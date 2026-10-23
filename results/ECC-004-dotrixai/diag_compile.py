"""ECC-004 diagnostic (before the PREREG): is the compiled baseline different from the eager baseline, or only chaotically diverged?

In the second calibration the compiled baseline ended 0.23 nats (mean of the last 20 of 60 updates) above the eager baseline, while
compiled and eager nGPT agreed to 0.001. This script separates a systematic difference from chaotic divergence:

  step    one update from identical weights on an identical batch, no chaos possible: loss, logits and gradients, compiled vs eager,
          next to the noise floor of eager vs eager with a different micro-batch split (fp32 and fp16, both arms)
  traj    200-update runs of the baseline (context 1024, lr 2e-3, fp16): eager and compiled, each with micro-batch 32 and 16, two runs
          at a time (one per GPU). Eager 32 vs eager 16 is the chaos floor; compiled vs eager at the same split is the compile effect.

    python diag_compile.py --data-dir DIR --out DIR [--parts step,traj]
Only the tuning validation windows are scored.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import ecc004 as E

HERE = Path(__file__).resolve().parent


def one_step(c, data, dev, compiled, split):
    torch.manual_seed(c.seed); m = E.LM(c).to(dev)
    xs, ys = data.batch(0); xs, ys = xs[:16].to(dev), ys[:16].to(dev)

    def loss_fn(x, y):
        with E.autocast(c, dev):
            logits = m(x)
        return F.cross_entropy(logits.float().reshape(-1, c.vocab), y.reshape(-1)), logits
    fn = torch.compile(loss_fn, backend=compiled) if compiled else loss_fn
    tot = 0.0; logits0 = None
    n = 16 // split
    for i in range(split):
        loss, logits = fn(xs[i * n:(i + 1) * n], ys[i * n:(i + 1) * n])
        (loss / split).backward(); tot += float(loss) / split
        if i == 0:
            logits0 = logits.detach().float()
    g = torch.cat([p.grad.detach().float().reshape(-1) for p in m.parameters()])
    return tot, logits0, g


def compare(a, b):
    la, xa, ga = a; lb, xb, gb = b
    return dict(loss_diff=lb - la, logits_max_abs_diff=float((xa - xb).abs().max()) if xa.shape == xb.shape else None,
                grad_rel_diff=float((ga - gb).norm() / ga.norm()), grad_cos=float(F.cosine_similarity(ga, gb, dim=0)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--parts', default='step,traj'); ap.add_argument('--backend', default='inductor'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.smoke:
        torch.set_num_threads(2)
    vocab = json.loads((Path(a.data_dir) / 'meta.json').read_text())['vocab']
    small = dict(d=64, layers=2, heads=2, hidden=256, ctx=128, tokens_per_update=128 * 16) if a.smoke else {}
    rep = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), env=E.env_info(dev), code_sha256=E.CODE_SHA, step=[], traj=[])
    save = lambda: E.save_json(out / 'diag.json', rep)
    log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)

    if 'step' in a.parts:
        for arm in ('baseline', 'ngpt'):
            for amp in ('fp32', 'fp16'):
                c = E.Config(arm=arm, vocab=vocab, amp=amp, seed=11, tokens=float(small.get('tokens_per_update', 131072)), **small)
                data = E.Data(a.data_dir, c, check_hashes=False)
                E._ROPE.clear(); eager = one_step(c, data, dev, None, 1)
                eager2 = one_step(c, data, dev, None, 2)
                torch._dynamo.reset(); comp = one_step(c, data, dev, a.backend, 1)
                r = dict(arm=arm, amp=amp, eager_loss=eager[0], noise_floor_eager_split2=compare(eager, eager2), compiled_vs_eager=compare(eager, comp))
                rep['step'].append(r); save()
                log(f'step {arm} {amp}: compiled vs eager {r["compiled_vs_eager"]}, floor {r["noise_floor_eager_split2"]}')

    if 'traj' in a.parts:
        # free this process's GPU memory first: in diag v1 (21 Okt) the step part still held it and both eager runs on GPU 0 hit OOM
        import gc
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        n_up = 200 if not a.smoke else 12
        ngpu = torch.cuda.device_count() if dev.type == 'cuda' else 1
        runs = [('none', 32), (a.backend, 32), ('none', 16), (a.backend, 16)] if not a.smoke else [('none', 16), (a.backend, 16), ('none', 8), (a.backend, 8)]
        for k in range(0, len(runs), max(1, ngpu)):
            procs = []
            for gi, (mode, mr) in enumerate(runs[k:k + max(1, ngpu)]):
                o = out / f'traj_{mode}_m{mr}'
                cmd = [sys.executable, str(HERE / 'ecc004.py'), 'train', '--arm', 'baseline', '--vocab', str(vocab), '--micro-rows', str(mr), '--lr', '2e-3',
                       '--tokens', str(n_up * (small.get('tokens_per_update', 131072))), '--eval-split', 'tune', '--eval-tokens', str(2 ** 16),
                       '--log-every', '1', '--compile', mode, '--data-dir', a.data_dir, '--out', str(o)]
                for key, v in small.items():
                    cmd += ['--' + key.replace('_', '-'), str(v)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gi)) if dev.type == 'cuda' else dict(os.environ, OMP_NUM_THREADS='2')
                if a.smoke:
                    cmd += ['--amp', 'fp32']                                   # laptop smoke test: light and fast
                procs.append((mode, mr, o, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))
            for mode, mr, o, p in procs:
                txt = p.communicate()[0]
                rec = json.loads((o / 'log.json').read_text()) if (o / 'log.json').exists() else None
                r = dict(mode=mode, micro_rows=mr, returncode=p.returncode, tail=txt[-400:] if p.returncode else '')
                if rec:
                    L = [x['loss'] for x in rec['curve']]
                    r.update(losses=L, tune_bits=[(q['update'], q['bits_per_token']) for q in rec['points']], skipped=rec['skipped_steps'])
                rep['traj'].append(r); save(); log(f'traj {mode} m{mr}: rc {p.returncode}')
            for f in out.glob('traj_*/ckpt.pt'):
                f.unlink()
        by = {(r['mode'], r['micro_rows']): r['losses'] for r in rep['traj'] if 'losses' in r}
        win = lambda L, a_, b_: sum(L[a_:b_]) / len(L[a_:b_])
        w = [(40, 60), (80, 100), (n_up - 20, n_up)] if not a.smoke else [(n_up - 6, n_up)]
        summary = {}
        for name, (x, y) in {'eager32_vs_eager16 (chaos floor)': (('none', runs[0][1]), ('none', runs[2][1])),
                             'compiled32_vs_eager32': (('none', runs[0][1]), (a.backend, runs[1][1])),
                             'compiled16_vs_eager16': (('none', runs[2][1]), (a.backend, runs[3][1])),
                             'compiled32_vs_compiled16': ((a.backend, runs[1][1]), (a.backend, runs[3][1]))}.items():
            if x in by and y in by:
                summary[name] = {f'mean loss diff, updates {a_ + 1}-{b_}': round(win(by[y], a_, b_) - win(by[x], a_, b_), 5) for a_, b_ in w}
        rep['traj_summary'] = summary; save(); log(json.dumps(summary, indent=1))
    rep['seconds'] = round(time.time() - t0, 1); save(); log('done')


if __name__ == '__main__':
    main()
