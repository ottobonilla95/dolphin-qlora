# I made my model worse. Then I got it to 88.89%.

### Fine-tuning Dolphin 8B on a Mac, with both experiments left in the open.

![Final accuracy: 88.89 percent](docs/assets/badge-accuracy.svg)
![Gain over base: 18.18 percentage points](docs/assets/badge-gain.svg)
![Training examples: 1080](docs/assets/badge-samples.svg)
![Runtime: Apple Silicon and MLX](docs/assets/badge-runtime.svg)

I wanted to teach an existing model to sort customer support messages into **27 intents**. My first QLoRA run made it worse—even while the training metrics looked better. A fresh run with a **10× lower learning rate** and checks of actual answers beat the original model.

![Three evaluations on the same 297 classification questions: base 70.71%, first adapter 54.21%, lower-rate adapter selected at step 450 88.89%.](docs/assets/results.png)

**54 more correct answers than the base.** The successful adapter scored **264/297**, up from **210/297**. All three sampled general answers also stayed coherent; the first adapter had made them repetitive.

[Read the experiment →](EXPERIMENT.md) · [Inspect the predictions →](results/20261028T115822561383Z.txt) · [Reproduce it →](REPRODUCING.md)

## What did the model learn?

Given a customer message, return a single label in JSON:

```text
“Where is my order?”  →  {"intent":"track_order"}
```

The base is **Dolphin3.0 Llama3.1 8B**, converted to 4-bit MLX. QLoRA trains a small, separate adapter while keeping the base weights frozen. This setup trained about **0.065%** of the model's parameters.

## What changed?

| | First run | Successful follow-up |
| --- | --- | --- |
| Learning rate | `0.0001` | **`0.00001`** |
| Training examples | 1,080 | Same 1,080 |
| Batch size / optimizer / adapter structure | 2 / Adam / last 8 blocks | Same |
| Checks of generated answers | After training | **Every 50 updates** |
| Checkpoint evaluated on the final suite | Final step 540 | **Step 450, chosen on validation** |
| Final classification accuracy | 54.21% | **88.89%** |

Before the full follow-up, a separate 27-example check confirmed that the setup could learn its own training examples. The full run then started fresh from the base, using all 1,080 examples.

**The lesson:** look at what the model actually answers. Lower token loss alone missed a serious regression. Gentler updates and validation-based checkpoint selection worked better here; we haven't separated how much each contributed.

## All on one laptop

| Model | Machine | Reported peak MLX memory | Monitored follow-up |
| --- | --- | --- | --- |
| Dolphin 8B, 4-bit | Apple M5 Pro, 48 GiB | ~6.20 GB | ~15.2 min, including validation |

This is one local learning experiment with one seed and a public synthetic dataset. The final suite had already been evaluated in run 001; **88.89% is a result on these 297 questions**, not a promise for every customer message. The three general probes are a small qualitative check.

## Explore the evidence

- **[Experiment in a few minutes](EXPERIMENT.md)** — the failure, the fix, the validation curve, and the limits.
- **[Reproduction guide](REPRODUCING.md)** — setup, training, evaluation, and trying a prompt.
- **[Run 002 comparison](results/training-002-comparison.json)** — scores, gains, losses, and integrity checks.
- **[Full technical record](docs/technical-record.md)** — exact settings, dataset provenance, and the diagnosis.

<details>
<summary><strong>Files, verification, and data terms</strong></summary>

| Location | Contents |
| --- | --- |
| `results/` | Predictions, scores, comparisons, and diagnostics |
| `data/` | Fixed training and validation splits, with provenance |
| `experiments/` | Follow-up configs, checkpoint checks, and summaries |
| `baseline-large-prompts.json` | 297 final classifications and 3 general probes |
| `run_experiment.py` | Tiny-fit and full-data experiments with monitoring |
| `train-config.yaml` | Original run's training settings |

Verify the first run's recorded evidence with Python's standard library:

```bash
python3 verify_results.py
```

The model and adapter weights remain local and are excluded from Git. Historical records retain their original absolute paths; see the reproduction guide for preparing another workspace.

Data is adapted from Bitext's public customer support dataset. Attribution, transformations, and data terms are in [the data notice](data/NOTICE.txt). Model weights are obtained separately and remain subject to their upstream terms. No blanket license for the original code or prose is added here.

</details>
