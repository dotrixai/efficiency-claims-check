# S-3: Reproducibility

Draft v0.1. A check here should be easy to verify, easy to run again, and easy to build on. Levels 1, 2 and 4 are required for every check. Level 3 is the goal.

## 1. Verifiable

Anyone can recompute the result from what is published, without a GPU.

- The PREREG is frozen, with its commit hash and time.
- Raw logs are published per run as JSONL, one line per evaluation point. RESULT lists the SHA-256 of every log.
- A standalone `verify.py` recomputes the cost-to-quality curve, the ratios and the verdict from those logs only. It runs on a CPU in under a minute. Its output must match RESULT exactly.

## 2. Re-runnable

Anyone can run one seed again and know what "the same" means.

- One notebook (or one command, for the CPU harness) runs one seed from start to end.
- Versions are pinned: container or image, PyTorch, CUDA, tokenizer.
- Data preparation is deterministic, and the SHA-256 of every token file is published.
- A written tolerance says how close a rerun must come, for example in final bits and R_FLOP. It comes from the measured spread between seeds, not a guess.
- The GPU hours for one seed are stated.
- `REPRODUCE.md` explains in plain words what is tested, how to run one seed, and what each number means.

## 3. Replicated (the goal)

Someone else runs it again and lands inside the tolerance. This is the REPLICATED level in [`CHARTER.md`](../CHARTER.md) §5. It cannot be forced, only made easy by level 2.

## 4. Buildable

Others can use the result in their own work, including as input to other research and for learning.

- A `result.json` with a fixed schema: claim ID, check ID, scale, hardware, seeds, Q, R_FLOP, R_time and R_steps per seed and per curve point, the verdict, the evidence level, and a MEASURED, ESTIMATED or PROJECTED label on each number. `verify.py` writes it.
- A draft technique note (`techniques/T###-...md`): when to use the technique, when not to, at what scale, with links to the evidence.

## Notes

- Checks published before this standard ([ECC-001](../results/ECC-001-dotrixai/), [ECC-002](../results/ECC-002-shikhar/)) do not meet it yet. They will be brought up to it later.
- Comments and changes go through a pull request, as in [`CHARTER.md`](../CHARTER.md) §11.
