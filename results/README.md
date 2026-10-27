# Results

One folder per check, named `ECC-###-<handle>` (the number is assigned by a maintainer), holding `PREREG.md`, `RESULT.md`, `REVIEW.md`, code and logs. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

| check | claim | by | status |
|---|---|---|---|
| [ECC-001](ECC-001-dotrixai/) | C12 nGPT, 4x to 20x fewer training steps | DotrixAI | **PARTIAL** at 0.73M params, laptop CPU: 1.47x fewer steps against the claimed 4x to 20x (2026-10-17) |
| [ECC-002](ECC-002-shikhar/) | C13 dynamic short convolutions | Shikhar | **EXPLORATORY**: R_FLOP 0.56 against the simple reference baseline, R_time 0.97; not publicly pre-registered (2026-10-18) |
| [ECC-003](ECC-003-AlphaCoderJay/) | C27 successive halving vs random search | AlphaCoderJay | **PRE-REGISTERED**, frozen 2026-10-27 (merge `b33910b`); laptop CPU runs not started |
