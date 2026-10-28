# Dolphin Classification Experiment and Diagnosis

## Purpose and outcome

The experiment tested whether local QLoRA fine tuning could improve an existing instruction model at classifying customer support messages. It also exercised a complete workflow: preserve the original model, establish a baseline, reserve evaluation examples, train a separate adapter, and compare generated answers.

The first adapter was unsuccessful. Classification accuracy fell from 70.71% to 54.21% on the fixed MLX evaluation, and the three general probes produced repetitive text. The cause of the regression has not been isolated.

## Model and hardware

The original Bionic model was `Dolphin3.0-Llama3.1-8B-Q4_K_S.gguf`. A copy was preserved with a separate inode and a matching SHA256 checksum. Bionic continued using the original model.

Training used a separate model converted from [dphn/Dolphin3.0-Llama3.1-8B](https://huggingface.co/dphn/Dolphin3.0-Llama3.1-8B), pinned at revision `f065677950dfc7e708d518d64cf1f5041ee007a0`. MLX conversion used 4-bit affine quantization with group size 64. This is a different quantization format from GGUF Q4_K_S, so the exact MLX model was evaluated before training.

The tokenizer vocabulary, token IDs, chat template, and EOS token were preserved during conversion. The GGUF and Hugging Face chat templates also matched. The EOS token was `<|im_end|>`, ID 128256.

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

The API and MLX runs explicitly supplied the saved system prompts. They did not inherit a particular Bionic conversation's system prompt. The before and after MLX runs had identical requests and suite hashes.

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

The saved adapter reproduced the final validation loss to within approximately 1.5e-8 on reload. The base weights and original GGUF checksum still matched their recorded values. All 1,080 training conversation prefixes matched the corresponding generation prefixes.

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

The measured conclusion is that this adapter regressed on the selected tasks. Generated validation accuracy and general probes would have exposed the problem earlier than monitoring loss alone. The original model remains the preferred model for this experiment.

Overly aggressive updates are a plausible explanation, not an established cause. The learning rate was 1e-4, compared with the installed MLX LM default of 1e-5. Adapting every eligible projection in eight blocks may also have made broader changes than necessary. A narrow classification-only objective may have contributed to degraded general answers. None of those variables was varied in a controlled comparison. The default configuration is a reference, not proof of an appropriate learning rate for this task. See the [MLX example configuration](https://github.com/ml-explore/mlx-lm/blob/main/examples/lora_config.yaml) and the installed version pinned in the lock file.

The next proposed experiment starts from the frozen base with newly initialized adapters and changes only the learning rate to 1e-5. Generated validation accuracy, invalid-label rate, and general probes should be checked every 25 to 50 steps. Checkpoint choice should use validation data, with the fixed final suite reserved for reporting. If a narrower q_proj and v_proj adapter is tested, it should be a separate controlled experiment. A tiny training-set fit test and closer label review are further diagnostics if learning remains poor.

Only the first training run and read-only diagnostics were performed. The lower-learning-rate run has not been performed. A single seed, synthetic public source data, potentially ambiguous labels, strict formatting rules, and a small general probe set limit how far these results can be generalized.

## Evidence files

| Evidence | File |
| --- | --- |
| Original 30-case API run | [Raw results](results/20261028T000342574774Z.json) |
| Original 300-case API run | [Raw results](results/20261028T000935661961Z.json) |
| MLX before training | [Raw results](results/20261028T002422930075Z.json) |
| MLX after training | [Raw results](results/20261028T004036824182Z.json) |
| Case gains and losses | [Comparison](results/training-001-comparison.json) |
| Training settings and loss | [Summary](results/training-001-summary.json), [metrics](results/training-001-metrics.jsonl) |
| Checkpoint and token audits | [Diagnostics](results/training-001-diagnostics.json) |
| Adapter reload | [Reload check](results/training-001-reload-check.json) |
| Frozen weights | [Integrity check](results/training-001-integrity.json) |

