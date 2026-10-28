# Dolphin QLoRA Fine Tuning Study

A local learning experiment with Dolphin3.0 Llama3.1 8B on an Apple Silicon Mac. Small QLoRA adapters classify customer support requests into 27 intents. The first run regressed; a fresh adapter trained at a ten-times lower learning rate, with checkpoint selection using generated validation answers, improved the final evaluation score.

Experiment date: 28 October 2026. This repository preserves both runs, their raw predictions, and the diagnosis that motivated the follow-up.

## Results

| Model | Correct classifications | Accuracy |
| --- | ---: | ---: |
| MLX base before training | 210 / 297 | 70.71% |
| Adapter 001: learning rate 0.0001, final step 540 | 161 / 297 | 54.21% |
| Adapter 002: learning rate 0.00001, selected step 450 | 264 / 297 | **88.89%** |

All evaluations used the same MLX base weights, questions, system prompts, and generation settings. The selected lower-rate adapter gained **18.18 percentage points over the base**, fixing 62 previously failing cases while losing 8 previously passing ones. Its checkpoint was chosen using validation examples, before its final evaluation.

In run 001, validation token loss fell from **2.883 to 0.141** while classification degraded and all three general answers became repetitive. The follow-up first passed a 27-example training-set fit check, then used the original 1,080 training examples with a lower learning rate. Full validation accuracy improved from **75.56% to 92.96%**. All three final general probes were coherent with the selected adapter. These are qualitative examples, not a broad general-capability score.

The frozen MLX base weights and original adapter were unchanged. All adapters remain separate from the base; their weights are excluded from Git. The selected follow-up adapter is available locally at `experiments/training-002-lr1e5/selected-adapter/`.

## Findings and conclusions

- Run 001 showed that a low token loss can coexist with incorrect generated labels and repetitive general answers.
- A tiny fit diagnostic reached 27/27 training examples at step 100 with the lower learning rate; this checks memorization, not generalization.
- The full follow-up changed the learning rate while preserving the training data, optimizer, adapter structure, and seed. Generated validation checks selected step 450 rather than the final step 540.
- This supports using gentler updates and generated-answer monitoring for this setup. A single seed and the previously used public evaluation suite do not establish an optimum or broad generalization.

Read the [experiment report](EXPERIMENT.md) for the method, evidence, limitations, and proposed next experiment. Read [reproduction instructions](REPRODUCING.md) to inspect the records or prepare a fresh run.

## Repository contents

| Location | Contents |
| --- | --- |
| `results/` | Full responses, scores, loss records, comparisons, and diagnostics |
| `data/` | Fixed training and validation splits, provenance, and tokenization check |
| `baseline-large-prompts.json` | Frozen 300-case evaluation suite: 297 scored classifications and 3 general probes |
| `training-model/*.json` | Model provenance and tokenizer compatibility checks |
| `adapters/intent-qlora-001/adapter_config.json` | Recorded adapter configuration |
| `train-config.yaml` | Original training configuration |
| `experiments/` | Follow-up configurations, validation responses, selection records, and summaries |
| `run_experiment.py` | Fresh tiny-fit and full-data runs with generated-answer monitoring |
| `*.py` | Preparation, training, inference, evaluation, and diagnostic scripts |

Historical records retain their original absolute local paths and checksums. Repository copies of those records have not been rewritten to make the paths portable. `make_workspace.py` prepares a separate folder with paths adjusted for a new run.

## Inspect the evidence

The following check verifies the first run's recorded evidence using Python's standard library, with no model download or GPU:

```bash
python3 verify_results.py
```

Follow-up evidence is summarized in [the run 002 comparison](results/training-002-comparison.json), with [raw final responses](results/20261028T115822561383Z.txt) and [the full experiment report](EXPERIMENT.md#follow-up-lower-learning-rate-with-generated-answer-monitoring).

The data includes modified subsets of Bitext's public customer support dataset. Attribution, the pinned source, transformations, and the applicable data license are recorded in [the data notice](data/NOTICE.txt). Model weights are obtained separately from the pinned upstream model and remain subject to its terms. No blanket license for the repository's original code or prose is added here.

