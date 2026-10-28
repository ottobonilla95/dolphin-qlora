# Dolphin QLoRA Fine Tuning Study

A local learning experiment with Dolphin3.0 Llama3.1 8B on an Apple Silicon Mac. A small QLoRA adapter was trained to classify customer support requests into 27 intents. The adapter reduced classification accuracy and damaged the three sampled general answers, despite a substantial reduction in validation token loss.

Experiment date: 28 October 2026. This repository preserves an unsuccessful experiment so its outcome and reasoning can be inspected.

## Results

| Model | Correct classifications | Accuracy |
| --- | ---: | ---: |
| MLX base before training | 210 / 297 | 70.71% |
| Same MLX base with adapter | 161 / 297 | 54.21% |

The comparison is **the same MLX model before versus after fine tuning: a decrease of 16.50 percentage points**. Both runs used identical questions, system prompts, and requested generation settings.

Validation token loss fell from **2.883 to 0.141**. General answers became repetitive on all three probes. Lower token loss did not mean better generated answers in this run.

The frozen MLX base weights were unchanged. The approximately 21 MB adapter remains a separate, unsuccessful candidate and is excluded from Git along with the model weights.

## Findings and conclusions

- Performance degradation appeared by the first saved checkpoint at step 180.
- Token inspection showed strong confidence in repeated JSON syntax alongside incorrect intent decisions.
- Tokenizer preservation, matching prompt prefixes, saved adapter reload, and unchanged base weights were checked. Three audited answers also reproduced without the generation cache.
- Overly aggressive updates are a **hypothesis**. The learning rate was ten times the installed MLX default, and adapters covered every eligible linear projection in the final eight blocks. Neither choice has been isolated as the cause.

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
| `*.py` | Preparation, training, inference, evaluation, and diagnostic scripts |

Historical records retain their original absolute local paths and checksums. Repository copies of those records have not been rewritten to make the paths portable. `make_workspace.py` prepares a separate folder with paths adjusted for a new run.

## Inspect the evidence

The following check uses Python's standard library and requires no model download or GPU:

```bash
python3 verify_results.py
```

The data includes modified subsets of Bitext's public customer support dataset. Attribution, the pinned source, transformations, and the applicable data license are recorded in [the data notice](data/NOTICE.txt). Model weights are obtained separately from the pinned upstream model and remain subject to its terms. No blanket license for the repository's original code or prose is added here.

