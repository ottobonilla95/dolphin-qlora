# Dolphin Classification Experiment and Diagnosis

## Purpose and outcome

The experiment tested whether local QLoRA fine tuning could improve an existing instruction model at classifying customer support messages. It also exercised a complete workflow: establish a baseline on the exact MLX training model, reserve evaluation examples, train a separate adapter, and compare generated answers.

The first adapter was unsuccessful. Classification accuracy fell from 70.71% to 54.21% on the fixed MLX evaluation, and the three general probes produced repetitive text. The cause of the regression has not been isolated.

## Model and hardware

Training used a model converted from [dphn/Dolphin3.0-Llama3.1-8B](https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B), pinned at revision `f065677950dfc7e708d518d64cf1f5041ee007a0`. MLX conversion used 4-bit affine quantization with group size 64. That exact quantized MLX model was evaluated before training and again with the trained adapter.

The tokenizer vocabulary, token IDs, chat template, and EOS token were preserved during conversion from the Hugging Face model. The EOS token was `<|im_end|>`, ID 128256.

The run used an Apple M5 Pro with 48 GiB unified memory, Python 3.13.15, MLX 0.32.4, MLX Metal 0.32.4, and MLX LM 0.32.2. Exact dependencies are in `requirements-lock.txt`.

## Task and data

Each classification prompt requested exactly one JSON object with one key, `intent`, chosen from 27 allowed values. The assistant training target was a compact object such as `{"intent":"cancel_order"}`. The training targets were intent labels rather than the upstream dataset's longer customer support answers.

The source was [Bitext customer support data](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), pinned at revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`. Its CSV contained 26,872 examples across 27 intents.

| Split | Examples | Distribution |
| --- | ---: | --- |
| Training | 1,080 | 40 per intent |
| Validation | 270 | 10 per intent |
| Final evaluation | 300 | 270 Bitext classifications, 27 hand-written classifications, 3 general probes |

The scored evaluation denominator is **297**, because the three general probes were reviewed qualitatively. Selection used seed 42. Exact normalized duplicates and lexical near duplicates were excluded between selected splits under the policy in `data/split-manifest.json`. Conflicting normalized labels and W-tagged examples were excluded during selection. The lexical policy cannot rule out every semantic paraphrase or the model's prior exposure to public data.

The suite SHA256 is `acb762b389cd6bac8a1e2f850c2edc779e25235dd3cea3c7648000ccaaae4a95`.

## Evaluation method

Each example used a fresh conversation with temperature 0 and seed 42. Classification allowed 128 output tokens; general probes allowed 256. No forced JSON schema was used. A pass required the entire response to parse as JSON, contain exactly the key `intent`, and match the expected allowed label. A correct label inside a code fence therefore failed the strict check.

Both MLX runs explicitly supplied the saved system prompts. The before and after runs had identical requests and suite hashes.

The general system prompt was `You are Dolphin, a helpful AI assistant.` The classification system prompt, allowed labels, and all messages are retained in the suite and raw results.

## Training configuration

| Setting | Value |
| --- | --- |
| Method | QLoRA: frozen quantized base with trainable LoRA matrices |
| Existing blocks adapted | Last 8 transformer blocks; no new transformer blocks added |
| Projections adapted | All eligible linear projections within those blocks |
| Rank and scale | Rank 8, MLX scale 20, dropout 0 |
| Trainable parameters | 5.243 million, approximately 0.065% of the model |
| Optimizer and learning rate | Adam, constant 0.0001, no warmup |
| Batch and accumulation | 2 examples, 1 accumulation step |
| Updates | 540, one pass through 1,080 examples |
| Maximum sequence length | 256 tokens; the longest checked training example was 167 |
| Prompt masking | Enabled; the final assistant message was the intended training target |
| Checkpoints | Steps 180, 360, and 540 |

Training took 472.68 seconds, approximately 7.9 minutes. Reported peak MLX memory was about 6.20 GB. Final adapter size was 20,983,592 bytes. These are measurements from this machine and run.

## Measured results

| Metric | MLX before | MLX after |
| --- | ---: | ---: |
| Correct intent and schema | 210 / 297 | 161 / 297 |
| Accuracy | 70.71% | 54.21% |
| Parseable JSON responses | 293 / 297 | 297 / 297 |
| Allowed schema and label | 273 / 297 | 246 / 297 |
| Request errors | 0 | 0 |

The adapter gained 29 previously failing classifications but lost 78 previously passing ones, for a net loss of 49. Its JSON became consistently parseable while its label choices and allowed-label compliance worsened.

The three general questions covered a Python dictionary explanation, simple arithmetic, and rewriting a customer message. The base answers were coherent. All three adapter answers became repetitive and unusable. Three probes establish a failure on those examples, not a broad numerical score of general capabilities.

Validation loss decreased from 2.882975 before training to 0.141368 after the final update. Intermediate validation losses were approximately 0.908, 0.340, and 0.140 after 179, 359, and 539 updates. These validations ran immediately before the corresponding checkpoint iteration's update; the checkpoint weights were saved after that update.

## Investigation after the regression

The saved adapter reproduced the final validation loss to within approximately 1.5e-8 on reload. The frozen MLX base weights still matched their recorded checksum. All 1,080 training conversation prefixes matched the corresponding generation prefixes.

A diagnostic subset selected the first validation example per intent. These 27 examples are separate from the frozen final evaluation:

| Saved checkpoint | Correct on the diagnostic subset |
| --- | ---: |
| Base, step 0 | 17 / 27 |
| Step 180 | 0 / 27 |
| Step 360 | 10 / 27 |
| Step 540 | 13 / 27 |

General probes were already repetitive at step 180. The degradation therefore appeared early, followed by a partial recovery. Sampled training examples also produced wrong labels, so ordinary memorization of training examples with poor transfer does not sufficiently explain the behavior.

For one validation cancellation example, a token audit showed that `cancel` had only 2.495% probability, while the model preferred `get`. Most surrounding JSON tokens had probabilities near 100%. The generated answer was `{"intent":"get_fee_methods"}`. A tracking example similarly favored `get` over `track`.

Loss averages across token predictions. Moreover, teacher forcing supplies the correct preceding answer tokens when calculating each prediction. Once the correct `cancel` token is supplied, predicting `_order` can be easy even though free generation never selected `cancel`. This explains how a favorable token loss can coexist with bad generated classifications. It does not identify why training damaged the label decisions.

Greedy decoding without a KV cache reproduced the same answers for the three audited examples. Those checks do not point to a cache-specific generation problem.

## Conclusions and remaining hypotheses

The measured conclusion is that adapter 001 regressed on the selected tasks. Generated validation accuracy and general probes would have exposed the problem earlier than monitoring loss alone. The MLX base is preferred over adapter 001; the separate follow-up below evaluates a new candidate.

Overly aggressive updates are a plausible explanation, not an established cause. The learning rate was 1e-4, compared with the installed MLX LM default of 1e-5. Adapting every eligible projection in eight blocks may also have made broader changes than necessary. A narrow classification-only objective may have contributed to degraded general answers. None of those variables was varied in a controlled comparison. The default configuration is a reference, not proof of an appropriate learning rate for this task. See the [MLX example configuration](https://github.com/ml-explore/mlx-lm/blob/main/examples/lora_config.yaml) and the installed version pinned in the lock file.

This diagnosis motivated a fresh adapter at learning rate 1e-5, with generated validation accuracy, invalid-label rate, and general probes checked every 50 steps. Checkpoint choice uses validation data, with the fixed final suite reserved for reporting. A narrower q_proj and v_proj adapter remains a possible separate experiment; it has not been tested here.

The first run and its diagnostics are preserved above. The lower-learning-rate follow-up is recorded below. A single seed, synthetic public source data, potentially ambiguous labels, strict formatting rules, and a small general probe set limit how far these results can be generalized.

## Evidence files

| Evidence | File |
| --- | --- |
| MLX before training | [Raw results](results/20261028T002422930075Z.json) |
| MLX after training | [Raw results](results/20261028T004036824182Z.json) |
| Case gains and losses | [Comparison](results/training-001-comparison.json) |
| Training settings and loss | [Summary](results/training-001-summary.json), [metrics](results/training-001-metrics.jsonl) |
| Checkpoint and token audits | [Diagnostics](results/training-001-diagnostics.json) |
| Adapter reload | [Reload check](results/training-001-reload-check.json) |
| Frozen weights | [Integrity check](results/training-001-integrity.json) |

## Follow-up: lower learning rate with generated-answer monitoring

On 28 October 2026, two fresh adapters were initialized from the unchanged MLX base. Neither resumed the failed first adapter. The new runner is `run_experiment.py`; all original model files, dataset splits, the reserved suite, and adapter 001 weights passed checksum verification before and after each successful run.

### Training-set fit diagnostic

The small diagnostic used 27 reviewed examples drawn from the original training split, one per intent, with learning rate `0.00001` and batch size 1. The base answered 23/27 correctly. The fresh adapter reached 26/27 at step 50 and 27/27 at step 100, triggering an early stop. The three development general answers remained coherent. This establishes that the setup can learn its training examples; it does not measure generalization. The complete diagnostic took 78.96 seconds, including generation checks.

The first monitoring attempt failed before any optimizer updates because its random-state restoration used an unsupported MLX API. The corrected monitor uses pure greedy generation without reseeding, verifies that the MLX random state is unchanged, preserves NumPy's state, and restores the model's training mode. This preparation failure is retained separately as `experiments/training-002-fit/`; the successful diagnostic is `experiments/training-002-fit-r2/`.

### Full-data experiment

The full run used the same 1,080 training examples, batch size 2, 540 updates, Adam optimizer, seed 42, last eight adapted blocks, rank 8, scale 20, and prompt masking as run 001. The learning rate changed from `0.0001` to `0.00001`. Reporting and checkpoint saving changed to every 50 updates; the full validation-loss schedule remained unchanged. No training examples were added or relabeled.

Generation checks used the first validation example per intent, the same fixed 27-example subset used in the earlier diagnostics. Three new general development prompts checked a Python list explanation, arithmetic, and a polite rewrite. They were reviewed qualitatively and were separate from the reserved final-suite probes.

| Completed updates | Correct on the 27-example validation subset |
| --- | ---: |
| Base | 17 |
| 50 | 19 |
| 100 | 22 |
| 150 | 21 |
| 200 | 24 |
| 250 | 23 |
| 300 | 25 |
| 350 | 25 |
| 400 | 25 |
| 450 | 26 |
| 500 | 26 |
| 540 | 26 |

The earliest checkpoint on a tie was selected, so the chosen adapter is **step 450**, after 900 training examples. Selection used this validation subset only. The selected checkpoint was then evaluated on all 270 validation examples with the same greedy decoding and strict schema checks as the base:

| Full validation measure | Base | Selected step 450 |
| --- | ---: | ---: |
| Correct classifications | 204 / 270 | 251 / 270 |
| Accuracy | 75.56% | 92.96% |
| Invalid schema or label | 24 | 1 |

The gain was 47 classifications, or 17.40 percentage points. All three development general responses remained coherent, including the correct arithmetic answer. These checks do not establish broad preservation of general capabilities.

The monitored full experiment took 909.33 seconds (about 15.2 minutes), including baseline generation, training, checkpoints, and full validation of the selected adapter. Peak reported MLX memory was approximately 6.20 GB. These timings include evaluation overhead and are not a training-only benchmark.

The selected weights are exported locally to `experiments/training-002-lr1e5/selected-adapter/`; weights remain excluded from Git. The final step-540 weights are retained separately. See the [full run summary](experiments/training-002-lr1e5/summary.json), [all generated checkpoint checks](experiments/training-002-lr1e5/monitor.json), [loss records](experiments/training-002-lr1e5/metrics.jsonl), and [tiny-fit summary](experiments/training-002-fit-r2/summary.json).

### Final reporting evaluation

After selecting step 450, the exported adapter was reloaded in a separate process and evaluated once on the original 300-case suite. The scored denominator remains 297 classifications; the remaining three cases are qualitative general probes. Checkpoint selection and training did not use these final classification examples. The suite was already evaluated in run 001, so this is an exploratory comparison on a previously used test set, not a newly collected blind benchmark.

| Final suite measure | Base | Adapter 001, step 540 | Adapter 002, selected step 450 |
| --- | ---: | ---: | ---: |
| Correct classifications | 210 / 297 | 161 / 297 | 264 / 297 |
| Accuracy | 70.71% | 54.21% | **88.89%** |
| Parseable JSON | 293 / 297 | 297 / 297 | 297 / 297 |
| Allowed schema and label | 273 / 297 | 246 / 297 | 295 / 297 |

Adapter 002 gained 62 base-model failures and lost 8 base-model successes: a net gain of 54 classifications, or **18.18 percentage points over the base**. There were no request errors. The three general answers were coherent: a valid dictionary explanation and example, correct arithmetic (8 euro change), and a polite rewrite within the requested 30-word limit. This limited check does not establish broad preservation of the base model's general abilities.

The final suite hash, every conversation and generation setting, and the selected adapter checksum were verified against the recorded base evaluation and selected checkpoint. The relative versus absolute model identifiers resolve to the same preserved directory. All 297 strict classification scores were independently recomputed from the final responses.

The result supports the lower-rate, validation-selected recipe for this dataset and model. It does not prove that `0.00001` is optimal, isolate all effects of checkpoint selection, or establish performance across seeds and other tasks. The next useful improvement would target remaining intent confusions using validation data and a new unseen evaluation set before further tuning against these results.

Evidence: [comparison and checks](results/training-002-comparison.json), [raw final result](results/20261028T115822561383Z.json), [readable responses and mismatches](results/20261028T115822561383Z.txt).

