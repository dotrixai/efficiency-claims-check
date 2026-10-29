# Method

Standard S-1, draft v1.1 (see [`standards/`](standards/)). Governed by [`CHARTER.md`](CHARTER.md).

How a check is measured and reported. The rules exist because each one caught a real mistake, often our own.

## 1. The quantity

For a target quality Q, let C(Q) be the training compute a recipe needs to first reach Q on held out text. The compute ratio of a method against a baseline is

    R(Q) = C_method(Q) / C_baseline(Q)

R below 1 means the method is cheaper at that quality. A claim of "2x less compute" is a claim of R = 0.5.

- **Q comes from the baseline.** Use a quality the baseline reaches near the end of its budget, not an early point. Early gaps often fade.
- **Report the curve.** Give R at every quality both runs pass, not only one point. Some methods help early and not late.
- **Interpolate, do not extrapolate.** C(Q) is read between measured evaluation points, linearly in log compute. If a run never reaches Q, report a bound and say so.

## 2. Compute

Report two numbers.

- **R_FLOP:** training FLOPs, counted for one forward and backward pass times the number of updates, plus any extra work such as table builds.
- **R_time:** measured wall clock or CPU time on the same hardware, in the same session, with nothing else running.

If R_FLOP and R_time disagree, report both and explain the gap. A method can save FLOPs and still cost more time.

## 3. Baseline

- Use a strong, well tuned recipe. For the CPU harness, the default is [ECC baseline v1](harness/baselines/). Another baseline must be argued in the pre-registration and accepted in review.
- Tune the baseline's learning rate with the same grid and the same effort as the method's.
- Include cheap published improvements in the baseline when they are standard for the scale. A method that only beats a weak baseline has not shown much.
- Use the same data, tokenizer, context length, batch size and evaluation text for both sides, unless the claim is about one of them.

## 4. Pre-registration

Before the first full training run, write down:

1. the claim being checked, with its source and claimed ratio;
2. the setup: model size, data, budget, baseline, tuning plan, seeds;
3. the target quality and how it is chosen;
4. the prediction, as numbers: your expected R and how sure you are;
5. the pass rule, for example "the claim holds at this scale if R_FLOP is at most 0.8 and R_time is at most 0.9";
6. the stop rules: when a run is ended early, and what that means.

Commit this file before training starts. The commit time is the freeze time. Do not edit it afterwards; write corrections in the result file.

Short pilot runs to check that code works are allowed before the freeze. Say how many and what they showed.

## 5. Checks

A run is invalid if any of these fail:

- **Causality.** The model must not see future tokens, and rows of a batch must not influence each other. The harness tests this by changing one token and checking that nothing before it, and nothing in the other row, changes.
- **No leakage.** No statistics built from the evaluation text.
- **Accounting.** All compute counts, including preprocessing and auxiliary tables.
- **Finite training.** No NaN or infinite loss.

## 6. Reporting

- Label every number: **MEASURED** (observed in a run), **ESTIMATED** (derived from measurements with a stated method) or **PROJECTED** (an extrapolation to an untested setting).
- State scale, seeds and hardware next to every result.
- A single seed is a single seed. Say so.
- Publish the result whatever it says. "Did not hold at 5M parameters on CPU" is a useful result.
- Do not generalize beyond the tested scale. A small scale check informs a claim; it rarely settles it.
- Make the check reproducible as set out in [S-3](standards/S-3-reproducibility.md): logs and a `verify.py` anyone can rerun, a pinned way to run one seed again, and a `result.json`.

## 7. Verdicts

| verdict | meaning |
|---|---|
| HOLDS | The pass rule was met at the tested scale. |
| PARTIAL | The method helps, but less than claimed, or only in FLOPs, or only early. |
| DOES NOT HOLD | The pass rule was not met at the tested scale. |
| INCONCLUSIVE | Noise, a bound or a failed check prevents a verdict. |

Every verdict carries its scale. The claim list keeps all verdicts per claim, not only the latest.
