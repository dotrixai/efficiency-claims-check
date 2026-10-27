# Efficiency Claims Check (ECC)

An open project to find out what really makes the training of AI models efficient, tested honestly and in public.

Many papers report that a new optimizer, architecture, schedule or data trick reaches the same quality with less compute. Some of these claims hold. Some shrink against a well tuned baseline or at a larger scale. Some measure FLOPs but not wall clock time. Practitioners rarely know which is which before they spend their own compute.

ECC does three things:

1. **Checks published claims** under fair, pre-registered conditions.
2. **Tests new ideas from anyone.** Propose your own technique or paper, and anyone can test it at any time under the same rules.
3. **Builds shared knowledge and standards:** verified results, practical techniques and a measurement method that practitioners and industry can rely on.

Every contribution follows the same rules, set out in the [**charter**](CHARTER.md):

1. **Strong baseline.** Compare against a well tuned modern recipe, not a weak default.
2. **Matched quality.** Report the compute each side needs to reach the same quality, not the loss at one step.
3. **FLOPs and time.** Report both. A FLOP saving that does not survive in measured time is reported as such.
4. **Pre-registration.** Write the prediction and the pass rule before training, and do not edit them afterwards.
5. **Independent review.** No one approves their own work.
6. **Negative results count.** A method that does not hold is a result, published with the same care.

Small scale cannot settle a claim made at a billion parameters. It can show where a claim depends on a weak baseline, a particular scale or an accounting choice. Every result states its scale next to its numbers.

## Who can contribute

Anyone, from any background in AI: researchers, engineers, students and independent practitioners. No affiliation or track record is needed, only that the work follows the [charter](CHARTER.md) and the [method](METHOD.md). Start with [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What is here

| path | what it is |
|---|---|
| [`CHARTER.md`](CHARTER.md) | The constitution: purpose, principles, kinds of contribution, evidence levels, review, roles, integrity, how rules change. |
| [`METHOD.md`](METHOD.md) | How a check is measured and reported (standard S-1). |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute each kind of work. |
| [`claims/`](claims/CLAIMS.md) | Published claims to check, with sources, claimed ratios and the independent evidence found. |
| [`proposals/`](proposals/) | Contributors' own ideas and techniques, written to be testable. |
| [`results/`](results/) | Checks, one folder each, with pre-registration, result, review, code and logs. |
| [`techniques/`](techniques/) | Practical technique notes, each with its evidence level. |
| [`standards/`](standards/) | Measurement and reporting rules proposed for wider adoption. |
| [`harness/`](harness/) | A CPU harness that trains a small language model under fixed rules, plus the [standard baselines](harness/baselines/). |
| [`templates/`](templates/) | Templates for pre-registrations, results, reviews, proposals and technique notes. |
| [`OPEN_AND_CLOSED.md`](OPEN_AND_CLOSED.md) | What is open here, and what DotrixAI keeps separate. |

## Results so far

| check | subject | by | verdict |
|---|---|---|---|
| [ECC-001](results/ECC-001-dotrixai/) | C12 nGPT: claimed 4x to 20x fewer steps | DotrixAI | **PARTIAL**: 1.47x at 0.73M parameters, and the gain fades during training |
| [ECC-002](results/ECC-002-shikhar/) | C13 dynamic short convolutions | Shikhar | **EXPLORATORY**: 1.8x in FLOPs, no time saving, weak baseline |
| [ECC-003](results/ECC-003-AlphaCoderJay/) | C27 successive halving: claimed about an order of magnitude faster search than random search | AlphaCoderJay | **PRE-REGISTERED**: frozen 2026-10-27, runs not started |

## The harness

`harness/` runs on a laptop CPU in about 8 minutes per run. A submission is one Python file that defines a model, an optimizer and a learning rate schedule. The harness owns the data order, the hidden evaluation text, the timing and the checks. Before any training it rejects code that looks at future tokens, couples rows of a batch, or reads files. It reports the compute ratio at matched quality, in FLOPs and in CPU time. Setup, including how to build the data files, is in [`harness/README.md`](harness/README.md).

The CPU harness fits optimizer, schedule and small architecture questions. Claims about data or large memories may need other setups, and a check may use any setup that follows [`METHOD.md`](METHOD.md).

## Who runs this

The project is stewarded by [DotrixAI](https://dotrixai.com), an independent AI research lab. Our own research program, CIR, tried to reach a strong optimized Transformer's quality with less training compute and did not succeed. We learned how easy it is to fool yourself with a weak baseline, a single scale or FLOPs read as time. Our public record, failures included, is at [dotrixai.com/cir](https://dotrixai.com/cir) and [github.com/dotrixai/cir-research](https://github.com/dotrixai/cir-research).

DotrixAI maintains the repository and reviews contributions under the charter. Credit for every piece of work goes to the people who did it.

## License

Code: MIT. Documentation and data: CC BY 4.0. See [`LICENSE`](LICENSE).
