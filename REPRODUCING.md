# Inspecting and Reproducing the Experiment

## Inspect the recorded outcome

Clone the repository and run the standard-library evidence check:

```bash
git clone git@github.com:ottobonilla95/dolphin-qlora.git
cd dolphin-qlora
python3 verify_results.py
```

The raw JSON results contain requests, responses, token usage, expected labels, and strict pass or fail judgments. The text files provide readable responses and mismatch lists. The report's numbers can be inspected without downloading any model.

Historical JSON and YAML records retain absolute paths from the original machine. Fields such as `training_started: false` in preparation manifests describe the time that manifest was created; `experiment-status.json` and the training records describe the completed experiment. The evaluation script now defaults to the MLX model and the full suite. The recorded MLX inference requests and responses are preserved unchanged.

## Prepare a separate workspace

The original training script refuses to overwrite its first-run logs and adapter folder. Create a new workspace rather than editing or deleting the evidence:

```bash
python3 make_workspace.py ../dolphin-qlora-reproduction
cd ../dolphin-qlora-reproduction
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

This requires an Apple Silicon Mac with the MLX Metal runtime for the recorded local setup. The original machine had 48 GiB unified memory; its reported peak MLX training allocation was about 6.20 GB. Those observations do not establish a minimum hardware requirement. Ensure there is space for approximately 16 GB of upstream model weights in the Hugging Face cache and the additional approximately 4.5 GB MLX model.

`make_workspace.py` copies the scripts and fixed datasets, rebases operational paths, and initializes new experiment state. It does not copy models, adapters, old results, or credentials, and it does not start training. It preserves the suite and training JSONL bytes. The copied split manifest's local evaluation path changes, so that manifest itself will have a different hash from the original record.

## Run the original configuration

The following commands download and convert the pinned model, measure a fresh baseline, and train one adapter. They reproduce the original experimental configuration, which regressed:

```bash
.venv/bin/python -u prepare_model.py
.venv/bin/python -u baseline.py --backend mlx --model ./training-model/dolphin-4bit --suite ./baseline-large-prompts.json --label mlx-before-training
.venv/bin/python -u train.py
.venv/bin/python -u baseline.py --backend mlx --model ./training-model/dolphin-4bit --adapter-path ./adapters/intent-qlora-001 --suite ./baseline-large-prompts.json --label mlx-after-training
.venv/bin/python -u diagnose.py
```

Model access and any acceptance of upstream model terms are handled by Hugging Face for your account. `uv` must be on PATH because the preparation script records the installed dependency freeze. Exact output equality across hardware, package builds, or runtime versions is not guaranteed by fixing a seed.

For a controlled lower-rate experiment in a separate workspace, change only `learning_rate` to `0.00001` in its generated `train-config.yaml`. The original `train.py` monitors token loss only. Changing its checkpoint frequency also requires adjusting `diagnose.py`, which expects steps 180, 360, and 540.

## Run monitored lower-rate experiments

With the existing local model and environment ready, the additional runner creates isolated experiment folders and refuses to reuse a name:

```bash
.venv/bin/python -u run_experiment.py --mode tiny --name my-fit-check
# Inspect experiments/my-fit-check/summary.json and monitor.json before proceeding.
.venv/bin/python -u run_experiment.py --mode full --name my-lower-rate-run
```

The tiny diagnostic uses 27 reviewed training examples, one per intent, at learning rate `0.00001` and batch size 1. It runs at most 270 updates and stops early if a generated-answer check gets all 27 right. This checks memorization of the training sample, not generalization.

The full mode uses the original 1,080 training examples, batch size 2, and 540 updates. The learning rate is reduced to `0.00001`; the optimizer, adapted layers, rank, scale, seed, prompt masking, and sequence length match run 001. Generated answers and checkpoints are recorded every 50 updates, including the final update. The original full validation-loss schedule is retained. The generation monitor preserves the trainer's random state and restores training mode after each check.

Checkpoint selection uses a fixed 27-example validation subset, choosing the earliest checkpoint on ties. The selected checkpoint is then measured on all 270 validation examples and compared against the base on those same examples. Three new general prompts are recorded for qualitative review; they are separate from the frozen final suite. The runner does not evaluate or select against the frozen final classification suite.

Each folder contains the effective configuration, raw generated responses, loss records, adapter checkpoint hashes, a summary, and original-file integrity checks. Adapter weights remain local and are excluded from Git. The original run's data, model, adapter, and results are preserved.

The selected checkpoint is exported to that experiment's `selected-adapter/` folder. Use it explicitly for an individual prompt:

```bash
.venv/bin/python predict.py --adapter-path experiments/my-lower-rate-run/selected-adapter 'Where is my order?'
```

Select configurations using validation data. Avoid repeatedly tuning against the fixed final suite; reserve new unseen examples if it has become part of the development loop.

## Try individual prompts

After preparing the model, the default helper uses the frozen base:

```bash
.venv/bin/python predict.py 'Where is my order?'
.venv/bin/python predict.py --general 'Explain a Python dictionary.'
```

After training, explicitly opt into the experimental adapter to compare it:

```bash
.venv/bin/python predict.py --adapter 'Where is my order?'
```

The recorded adapter and model binaries are excluded from Git, so a fresh clone cannot run their saved checkpoints until weights are obtained or a new run is completed. This helper uses MLX directly.

## Data provenance

The committed JSONL files are the exact prepared training and validation splits. The complete upstream CSV is excluded. Its pinned download URL, SHA256, and original row indices are in `data/split-manifest.json` and `baseline-large-prompts.json`. To rebuild splits from source, use a separate folder and download the CSV to `evaluation-source/` first; the builders refuse to overwrite existing split and suite outputs.

The source and transformations are described in [the data notice](data/NOTICE.txt). MLX model provenance and weight checksums are retained in the model manifest. The workflow uses the downloaded and converted MLX model throughout.

