# Reproduce the experiment

**To read the results, no model download or GPU is needed.** To run training, use an Apple Silicon Mac with MLX. The recorded experiments used an M5 Pro with 48 GiB memory and reported about 6.20 GB peak MLX memory; this is an observation, not a tested minimum requirement.

## Read or verify the saved results

Start with the [results guide](results/README.md) for scores, remaining errors, and links to the predictions. To check the first run's recorded scores and evidence:

```bash
git clone git@github.com:ottobonilla95/dolphin-qlora.git
cd dolphin-qlora
python3 verify_results.py
```

That command uses Python's standard library. It verifies run 001 only; it does not rerun inference. Run 002's recorded comparison is in [training-002-comparison.json](results/training-002-comparison.json).

## Prepare a new workspace

From the cloned repository, create a separate folder to keep the original results intact:

```bash
python3 make_workspace.py ../dolphin-qlora-reproduction
cd ../dolphin-qlora-reproduction
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -u prepare_model.py
```

`uv` must be installed and on PATH. Model access and upstream terms are handled through Hugging Face for the account being used. Allow disk space for roughly 16 GB of upstream weights in the Hugging Face cache and another 4.5 GB for the converted MLX model.

The workspace helper copies scripts and fixed datasets, updates local paths, and creates fresh experiment state. It does not copy model weights, adapters, old results, or credentials. The preparation command then downloads and converts the pinned model. A fixed seed aids comparison, but output can vary across hardware and package builds.

## Repeat the lower-rate experiment

Use a new name for each run; the runner refuses to overwrite an existing experiment folder.

```bash
# Check that a fresh adapter can learn 27 reviewed training examples.
.venv/bin/python -u run_experiment.py --mode tiny --name fit-check

# Review experiments/fit-check/summary.json and monitor.json.
# If the learning check passes, start another fresh adapter on all 1,080 examples.
.venv/bin/python -u run_experiment.py --mode full --name lower-rate-run
```

| Stage | Training setup | What it checks |
| --- | --- | --- |
| Small learning check | 27 examples, batch size 1, learning rate 0.00001 | Can the model learn those same examples? |
| Full experiment | 1,080 examples, batch size 2, learning rate 0.00001, 540 updates | Does accuracy improve on separate validation questions? |

The small check stops early if it answers all 27 training examples correctly, with a limit of 270 updates. This measures memorization. The full run keeps the original optimizer, adapter structure, seed, prompt masking, and sequence length.

Every 50 updates, the full run answers the same 27 validation questions and three general questions. The earliest checkpoint with the best monitored score is selected, then evaluated on all 270 validation examples. A **checkpoint** is a saved version of the adapter during training. The final test suite is kept out of checkpoint selection.

Each experiment folder records the configuration, answers, loss values, checkpoint hashes, and summary. The selected weights are exported to `selected-adapter/`. The original data, model, and adapter are checked for changes before and after training.

## Try an answer or run the final evaluation

Use the newly selected adapter explicitly:

```bash
.venv/bin/python predict.py --adapter-path experiments/lower-rate-run/selected-adapter 'Where is my order?'
.venv/bin/python predict.py --adapter-path experiments/lower-rate-run/selected-adapter --general 'Explain a Python dictionary.'
```

For comparison, omit `--adapter-path` to use the frozen base. The older `--adapter` flag loads run 001, which reduced accuracy.

After checkpoint selection, evaluate the chosen adapter on the final suite:

```bash
.venv/bin/python -u baseline.py --backend mlx --model ./training-model/dolphin-4bit --adapter-path ./experiments/lower-rate-run/selected-adapter --suite ./baseline-large-prompts.json --label lower-rate-selected
```

The runner records the base's full validation score automatically. For a fresh before/after comparison on the final suite, also evaluate the base with the same command, omitting `--adapter-path` and using a different `--label`.

Use validation data for further tuning. Collect a new unseen test set if the existing final-suite answers have become part of the development process.

<details>
<summary><strong>Repeat the original, unsuccessful configuration</strong></summary>

In another fresh workspace, after model preparation:

```bash
.venv/bin/python -u baseline.py --backend mlx --model ./training-model/dolphin-4bit --suite ./baseline-large-prompts.json --label mlx-before-training
.venv/bin/python -u train.py
.venv/bin/python -u baseline.py --backend mlx --model ./training-model/dolphin-4bit --adapter-path ./adapters/intent-qlora-001 --suite ./baseline-large-prompts.json --label mlx-after-training
.venv/bin/python -u diagnose.py
```

These commands use the original learning rate of `0.0001`. The original trainer tracks token loss; its diagnostic script expects checkpoints at steps 180, 360, and 540.

</details>

<details>
<summary><strong>Recorded paths, data sources, and model files</strong></summary>

Historical JSON and YAML files keep their original absolute paths. Preparation records describe their state when created; for example, `training_started: false` is a historical value, not the current outcome. The completed runs have their own training and evaluation summaries.

The committed training and validation splits retain their original bytes. The workspace helper changes the split manifest's local evaluation path, so its checksum changes even though the dataset does not.

The full source CSV is excluded from Git. Its pinned URL, checksum, and row indices are recorded in `data/split-manifest.json` and `baseline-large-prompts.json`. Attribution and data terms are in [the data notice](data/NOTICE.txt).

Model and adapter binaries are also excluded from Git. A fresh clone needs model preparation and training before the prediction commands can use an adapter. The model manifest records the source revision, conversion, and checksums.

</details>
