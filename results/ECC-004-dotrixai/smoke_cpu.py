"""ECC-004 smoke test on a laptop CPU: tiny models, synthetic tokens, a few minutes at most. Checks the code paths, not any claim.

    python smoke_cpu.py [--out DIR]

For both arms and for fp32, fp16 and bf16 autocast:
  - training runs and stays finite, and held-out bits fall;
  - the causality test passes on the initial and the trained model;
  - a run split into two segments (checkpoint, resume) ends at the same evaluation points as an uninterrupted run (fp32);
  - the torch.compile path (aot_eager backend on CPU) gives the same evaluation points as eager (fp32);
Also: the causality test catches a non-causal model and a batch-coupled model (negative controls), the sealed and tuning windows do not overlap, the token-file hashes
are enforced, and the analytic FLOP count is compared with torch.utils.flop_counter.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ecc004 as E

V = 512


def make_data(d):
    """A random first-order Markov chain over V tokens, so there is something to learn."""
    rng = np.random.default_rng(0)
    nxt = rng.integers(0, V, size=(V, 4))                                        # each token has 4 likely successors
    def chain(n, seed):
        r = np.random.default_rng(seed); out = np.empty(n, dtype=np.uint16); t = 0
        for i in range(n):
            t = int(nxt[t, r.integers(0, 4)]) if r.random() < 0.9 else int(r.integers(0, V)); out[i] = t
        return out
    d.mkdir(parents=True, exist_ok=True)
    (d / 'train.bin').write_bytes(chain(400_000, 1).tobytes()); (d / 'val.bin').write_bytes(chain(40_000, 2).tobytes())
    np.save(d / 'token_bytes.npy', np.full(V, 3, dtype=np.int32))
    sha = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ('train.bin', 'val.bin', 'token_bytes.npy')}
    (d / 'meta.json').write_text(json.dumps(dict(vocab=V, sha256=sha)))


def tiny(arm, amp, **kw):
    base = dict(arm=arm, d=64, layers=2, heads=2, hidden=256, vocab=V, ctx=64, tokens_per_update=64 * 8, micro_rows=4, tokens=64 * 8 * 60,
                lr=4e-3, amp=amp, seed=11, eval_tokens=64 * 16, eval_fracs=(0.1, 0.5, 1.0), log_every=10)
    base.update(kw)
    return E.Config(**base)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); a = ap.parse_args()
    torch.set_num_threads(2)
    root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix='ecc004_smoke_'))
    data = root / 'data'; make_data(data)
    cpu = torch.device('cpu'); report = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code_sha256=E.CODE_SHA, torch=torch.__version__,
                                             runs=[], checks={})
    fails = []

    def check(name, ok, detail):
        report['checks'][name] = dict(ok=bool(ok), detail=detail); print(('PASS ' if ok else 'FAIL ') + name, detail, flush=True)
        if not ok:
            fails.append(name)

    for arm in ('baseline', 'ngpt'):
        for amp in ('fp32', 'fp16', 'bf16'):
            c = tiny(arm, amp); t0 = time.time()
            rec = E.train(c, data, root / f'{arm}_{amp}', device=cpu, log=lambda s: None)
            pts = rec['points']
            row = dict(arm=arm, amp=amp, status=rec['status'], params=rec['params'], bits=[p['bits_per_token'] for p in pts],
                       skipped=rec['skipped_steps'], nonfinite=rec['nonfinite_losses'], train_time_s=round(rec['train_time_s'], 2),
                       wall_s=round(time.time() - t0, 1), causal_init=rec['checks']['causal_init'], causal_trained=rec['checks']['causal_trained'],
                       causal_init_amp=rec['checks']['causal_init_amp'], flops=rec['flops_per_token'])
            report['runs'].append(row)
            check(f'{arm}/{amp} trains, finite, bits fall', rec['status'] == 'done' and rec['nonfinite_losses'] == 0 and pts[-1]['bits_per_token'] < pts[0]['bits_per_token'] - 0.1,
                  dict(bits=row['bits'], skipped=row['skipped'], wall_s=row['wall_s']))
            check(f'{arm}/{amp} causal init+trained', rec['checks']['causal_init']['ok'] and rec['checks']['causal_trained']['ok'],
                  dict(init=rec['checks']['causal_init'], trained=rec['checks']['causal_trained']))

    # the compiled path (aot_eager backend: same graph capture as inductor, runs on a CPU without a C++ toolchain) matches eager
    for arm in ('baseline', 'ngpt'):
        eager = next(r for r in report['runs'] if r['arm'] == arm and r['amp'] == 'fp32')
        rec = E.train(tiny(arm, 'fp32', compile='aot_eager'), data, root / f'{arm}_compiled', device=cpu, log=lambda s: None)
        diff = max(abs(a - p['bits_per_token']) for a, p in zip(eager['bits'], rec['points']))
        check(f'{arm} compiled path matches eager (fp32)', rec['status'] == 'done' and diff < 1e-3, dict(max_bits_diff=diff))

    # resume equivalence (fp32): 60 updates straight vs 25 + resume 35
    for arm in ('baseline', 'ngpt'):
        c = tiny(arm, 'fp32')
        full = E.train(c, data, root / f'resume_full_{arm}', device=cpu, log=lambda s: None)
        part = E.train(c, data, root / f'resume_part_{arm}', device=cpu, log=lambda s: None, stop_at_update=25)
        rest = E.train(c, data, root / f'resume_part_{arm}', device=cpu, log=lambda s: None, resume=root / f'resume_part_{arm}' / 'ckpt.pt')
        diff = max(abs(p['bits_per_token'] - q['bits_per_token']) for p, q in zip(full['points'], rest['points']))
        check(f'{arm} resume matches uninterrupted run', part['status'].startswith('incomplete') and rest['status'] == 'done'
              and [p['update'] for p in full['points']] == [p['update'] for p in rest['points']] and diff < 1e-4,
              dict(max_bits_diff=diff, segments=len(rest['segments'])))

    # negative control: a non-causal model must fail the causality test
    orig = F.scaled_dot_product_attention
    try:
        E.F.scaled_dot_product_attention = lambda q, k, v, is_causal=False, scale=None: orig(q, k, v, is_causal=False, scale=scale)
        for arm in ('baseline', 'ngpt'):
            torch.manual_seed(0); m = E.LM(tiny(arm, 'fp32'))
            r = E.causality(m, tiny(arm, 'fp32'), cpu)
            check(f'{arm} non-causal model is rejected', not r['ok'], r)
    finally:
        E.F.scaled_dot_product_attention = orig

    # negative control: a model that mixes rows of the batch must fail the causality test
    class Coupled(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            z = self.m(x)
            return z + z.mean(0, keepdim=True)
    for arm in ('baseline', 'ngpt'):
        torch.manual_seed(0); m = Coupled(E.LM(tiny(arm, 'fp32')))
        r = E.causality(m, tiny(arm, 'fp32'), cpu)
        check(f'{arm} batch-coupled model is rejected', not r['ok'] and r['max_change_other_row'] > 0, r)

    # sealed and tuning windows are disjoint
    c = tiny('baseline', 'fp32'); D = E.Data(data, c)
    s, t = D.eval_windows('sealed'), D.eval_windows('tune')
    check('sealed and tuning windows disjoint', max(s) + c.ctx + 1 <= min(t), dict(sealed=[min(s), max(s)], tune=[min(t), max(t)], val=len(D.val)))

    # hash enforcement
    bad = root / 'data_bad'; shutil.copytree(data, bad)
    b = bytearray((bad / 'val.bin').read_bytes()); b[0] ^= 1; (bad / 'val.bin').write_bytes(bytes(b))
    try:
        E.Data(bad, c); ok = False
    except AssertionError:
        ok = True
    check('changed token file is refused', ok, {})

    # second epoch is refused
    try:
        E.Data(data, tiny('baseline', 'fp32', tokens=10 ** 7)); ok = False
    except AssertionError:
        ok = True
    check('budget beyond one epoch is refused', ok, {})

    # FLOPs: analytic vs counter at the realistic shape (forward and backward on one row, CPU)
    for arm in ('baseline', 'ngpt'):
        c = E.Config(arm=arm, ctx=256, amp='fp32'); torch.manual_seed(0); m = E.LM(c)
        an, fc = E.flops_per_token_analytic(c), E.flops_per_token_counter(m, c, cpu)
        report['checks'].setdefault('flops', {})[arm] = dict(params=E.n_params(m), analytic=an, flop_counter=fc, ratio=fc / an)
        print('flops', arm, E.n_params(m), an, fc, round(fc / an, 4), flush=True)

    report['fails'] = fails
    (root / 'smoke_report.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('report:', root / 'smoke_report.json'); print('ALL PASS' if not fails else f'{len(fails)} FAIL: {fails}')


if __name__ == '__main__':
    main()
