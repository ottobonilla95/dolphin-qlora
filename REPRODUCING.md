# Inspecting and Reproducing the Experiment

## Inspect the recorded outcome

Clone the repository and run the standard-library evidence check:

```bash
git clone git@github.com:ottobonilla95/dolphin-qlora-study.git
cd dolphin-qlora-study
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

For the proposed controlled experiment, use another fresh workspace and change only `learning_rate` to `0.00001` in its generated `train-config.yaml`. The recorded trainer monitors token loss; more frequent generated validation checks would require adding a monitoring callback or saving more checkpoints and evaluating them. Neither improvement was part of the original run. Changing checkpoint frequency also requires adjusting `diagnose.py`, which currently expects steps 180, 360, and 540.

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

