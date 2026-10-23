"""ECC-004 analysis: compute ratios at matched quality from ecc004.py logs (log.json).

Two estimators, both reported:

  curve     METHOD.md section 1, as in the harness and ECC-001: Q = the baseline's final bits per token; C_method(Q) is read off the
            evaluation points of ONE method run, interpolated linearly in log cost. The method's points before its own end are taken
            mid-schedule (learning rate not yet annealed), so this estimator is conservative for a method that crosses Q early.
  budgets   the paper's estimator: several method runs, each with its own full schedule over a smaller token budget; C_method(Q) is
            read off their FINAL points, interpolated linearly in log cost.

    python analyze.py --baseline B/log.json --method M/log.json [--budget-runs M1/log.json M2/log.json ...] [--key flops|train_time_s|tokens]
"""
import argparse
import json
import math


def ratio(points, q, c_ref, key):
    """C(q) / c_ref along points (dicts with key and bits_per_token); (value, bound): bound None, 'lower' (never reached q) or 'upper'
    (already below q at the first point). Same rule as harness/gym_harness.py ratio()."""
    pts = sorted(((p[key], p['bits_per_token']) for p in points), key=lambda t: t[0])
    if pts[-1][1] > q:
        return pts[-1][0] / c_ref, 'lower'
    if pts[0][1] <= q:
        return pts[0][0] / c_ref, 'upper'
    for (c0, b0), (c1, b1) in zip(pts, pts[1:]):
        if b0 > q >= b1:
            f = (b0 - q) / (b0 - b1)
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0))) / c_ref, None
    return pts[-1][0] / c_ref, 'lower'


def final_point(rec):
    assert rec['status'] == 'done', f'run not done: {rec["status"]}'
    return rec['points'][-1]


def curve_estimate(base, meth, key):
    q = final_point(base)['bits_per_token']; cref = final_point(base)[key]
    r, b = ratio(meth['points'], q, cref, key)
    curve = []
    for p in base['points']:
        rr, bb = ratio(meth['points'], p['bits_per_token'], p[key], key)
        curve.append(dict(baseline_update=p['update'], Q=p['bits_per_token'], R=round(rr, 4), bound=bb))
    return dict(Q=q, R=round(r, 4), bound=b, curve=curve)


def budget_estimate(base, runs, key):
    q = final_point(base)['bits_per_token']; cref = final_point(base)[key]
    finals = [final_point(r) for r in runs]
    r, b = ratio(finals, q, cref, key)
    return dict(Q=q, R=round(r, 4), bound=b, finals=[dict(tokens=p['tokens'], cost=p[key], bits_per_token=p['bits_per_token']) for p in finals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True); ap.add_argument('--method', required=True); ap.add_argument('--budget-runs', nargs='*', default=[])
    ap.add_argument('--key', default='flops', choices=['flops', 'train_time_s', 'tokens'])
    a = ap.parse_args()
    load = lambda p: json.loads(open(p, encoding='utf-8').read())
    base, meth = load(a.baseline), load(a.method)
    out = dict(key=a.key, curve=curve_estimate(base, meth, a.key))
    if a.budget_runs:
        out['budgets'] = budget_estimate(base, [meth] + [load(p) for p in a.budget_runs], a.key)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
