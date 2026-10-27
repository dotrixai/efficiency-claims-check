# Reproducing ECC-004

This folder tests one claim: that nGPT, a Transformer whose weights and hidden states are kept at unit length, reaches the same loss as a normal GPT with 4x fewer training steps at context 1024, and 10x fewer at 4096. The full plan is in `PREREG.md`. This page says how to check our numbers, and how to run a seed yourself.

## Check our numbers (CPU, seconds)

```
python verify.py --check
```

This reads only the raw logs in `logs/final/`. It recomputes every ratio and the verdict, and compares them with `result.json`. It prints `OK` or `MISMATCH`. Only the Python standard library is needed.

## What the numbers mean

- **Q:** the baseline's loss (bits per token on held-out text) at the end of its training. It is the target.
- **R_steps:** how many training steps nGPT needs to reach Q, as a fraction of the baseline's. R_steps = 0.25 means 4x fewer. The paper's claim is a claim about this number.
- **R_FLOP:** the same in compute. Both models do the same matrix work per step here, so R_FLOP equals R_steps.
- **R_time:** the same in GPU time. nGPT does extra normalization work in each step, so R_time is larger than R_steps.
- **Two ways to read R:**
  - main: train nGPT several times with smaller budgets (1/2, 1/4, 1/8 of the baseline's), each with its own learning-rate schedule, and see where the final losses cross Q;
  - also reported: read it off the curve of a single full-length nGPT run, the way ECC-001 did.
- **Bounds:** "≤" means nGPT was already better than Q at its smallest budget, and "≥" means it never got there.
- **Labels:** MEASURED means observed in a run. ESTIMATED means computed from measurements by a stated rule. R_time is ESTIMATED: R_steps times the measured time ratio per step.

## Run one seed yourself

You need a free Kaggle account with GPU access (phone verified).

1. **Data.** Run `prepare_data.py` as a Kaggle CPU notebook with internet on (about 30 minutes, no GPU). Check that the four SHA-256 values it prints match `PREREG.md`.
2. **One seed.** In a GPU T4 x2 notebook, with the data notebook's output attached, run:
   ```
   python session.py final --ctx 1024 --size 25m --seed 11 --lr-baseline <from result.json> --lr-ngpt <from result.json> --data-dir <data> --out /kaggle/working/s11
   ```
   This trains the baseline and nGPT at several budgets, then times both. It takes about 3.5 GPU hours at 25M / 1024, about 5.5 at 25M / 4096 and about 10.5 at 89M (`--size 89m`, 1024 only), PROJECTED from our calibration.
3. **Compare.** Copy the output folder into `logs/final/` and run `python verify.py`. Your seed should land within the tolerance in `PREREG.md`:
   - final bits per token within the range of our three seeds, plus 0.02;
   - R within the range of our three seeds, plus 0.05.

A ready Kaggle notebook link will be added here once the result is out.

## Tell us

Open an issue with your logs, whether your numbers match or not. A rerun that disagrees is as useful as one that agrees.
