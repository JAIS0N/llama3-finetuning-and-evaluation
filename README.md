# Fine-Tuning Llama-3.2-1B for Dialogue Summarization

A two-notebook project for **parameter-efficient fine-tuning** of Meta's Llama-3.2-1B-Instruct model on a dialogue summarization task, followed by rigorous quantitative evaluation using ROUGE and BERTScore metrics.

---

## Repository Structure

```
├── Fine_Tune_Llama_3_2_1B.ipynb         # Notebook 1 — Data prep, LoRA training, adapter saving
├── Evaluate_Fine_Tuned_Model.ipynb       # Notebook 2 — Load adapter, run inference, compute metrics
└── README.md
```

---

## Background & Motivation

Large language models (LLMs) are powerful out of the box, but their generic outputs often fall short on specialized tasks. **Fine-tuning** adapts a pre-trained model to a specific domain or format without training from scratch.

This project showcases:

- **4-bit quantization** (QLoRA) — loads a 1B-parameter model in ~1 GB of VRAM instead of ~4 GB.
- **LoRA (Low-Rank Adaptation)** — trains tiny adapter matrices (~1–2% of total parameters) instead of the full model, cutting training cost dramatically.
- **Completion-only training** — the model only learns to predict the *summary*, not the repeated prompt tokens, leading to cleaner gradients.
- **Automated evaluation** — ROUGE and BERTScore metrics give an objective view of improvement over the baseline.

---

## Dataset

| Property | Value |
|---|---|
| Name | `knkarthick/dialogsum` (Hugging Face Hub) |
| Split used for training | `train` |
| Split used for validation | `validation` |
| Split used for evaluation | `test` (200 random samples) |
| Each record contains | `id`, `topic`, `dialogue`, `summary` |

Every example is a real human–human conversation with a human-written reference summary and an associated topic tag (e.g., *shopping*, *medical*, *daily life*). The **topic** is fed into the prompt so the model can calibrate the tone of its summaries.

---

## Notebook 1 — `Fine_Tune_Llama_3_2_1B.ipynb`

### Purpose
Fine-tune Llama-3.2-1B-Instruct with LoRA adapters and save the best checkpoint.

### Step-by-step walkthrough

#### 1. Install Dependencies
```bash
pip install -q -U bitsandbytes transformers peft accelerate datasets trl
```
| Library | Role |
|---|---|
| `bitsandbytes` | 4-bit / 8-bit quantization kernels |
| `transformers` | Model loading, tokenizer, training arguments |
| `peft` | LoRA adapter creation and management |
| `accelerate` | Device placement and mixed precision |
| `datasets` | Dataset loading and processing |
| `trl` | `SFTTrainer` for supervised fine-tuning |

#### 2. GPU Setup
The notebook detects CUDA and prints device name, total VRAM, allocated memory, and reserved memory. Training requires a GPU; a free-tier Colab T4 (16 GB) or equivalent is sufficient.

#### 3. Load the Dataset
```python
dataset = load_dataset("knkarthick/dialogsum")
```
The `train`, `validation`, and `test` splits are all loaded in one call.

#### 4. Quantization Config (BitsAndBytes)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',          # NormalFloat4 — better than uniform 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
```
**NF4 quantization** maps weights to a 4-bit grid optimized for normally distributed values, which neural network weights typically follow. Computations still happen in `float16` internally.

#### 5. Load Base Model
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map={"": 0},          # pin everything to GPU 0
    quantization_config=bnb_config,
    trust_remote_code=True,
)
```
> **Note:** You need a Hugging Face account and must accept Meta's Llama licence before the model can be downloaded.

#### 6. Tokenizer Setup
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = '<|finetune_right_pad_id|>'
```
`padding_side="left"` is required for causal LMs so that the model always sees the rightmost (most recent) tokens during generation. A dedicated fine-tuning pad token avoids polluting the EOS token.

#### 7. Prompt Template
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert on summarizing conversations considering a particular topic.
The user request will contain the topic and the conversation
Answer with the summary only. Do not explain your answer
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Topic: {topic}
Conversation: {dialogue}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{summary}
```
This follows the **Llama 3 chat template** exactly. During training, `{summary}` is filled with the ground-truth reference. During inference it is left empty and the model generates it.

#### 8. Baseline Test
Before any fine-tuning, the notebook runs `generate_response()` on one training example and prints the model's output next to the human-written summary. This establishes the **pre-training baseline** you'll compare against after fine-tuning.

#### 9. Dataset Processing Pipeline
Three functions prepare the dataset:

| Function | What it does |
|---|---|
| `apply_prompt(sample)` | Formats each record into the full prompt string and appends an EOS token |
| `process_batch(batch, tokenizer, max_length)` | Tokenizes a batch of `text` fields |
| `process_dataset(dataset, tokenizer, max_length)` | Chains `apply_prompt` → `process_batch` → filters long sequences → shuffles |

Any sequence longer than `model.config.max_position_embeddings` is dropped at the filter step.

#### 10. LoRA Configuration
```python
lora_config = LoraConfig(
    r=32,                    # Rank — higher = more capacity but more parameters
    lora_alpha=32,           # Scaling factor (effective LR multiplier = alpha/r = 1)
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```
LoRA injects small trainable rank-32 matrices into the **query, key, value** projection layers and the **dense** (output) projection. Everything else stays frozen. The total number of *new* trainable parameters is typically < 20 million.

#### 11. Training Arguments
```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,    # Effective batch size = 4
    num_train_epochs=1,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",         # 8-bit Adam saves ~50% optimizer memory
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,
    load_best_model_at_end=True,      # Keeps the checkpoint with lowest eval loss
    save_total_limit=3,               # Only keep the 3 most recent checkpoints
    metric_for_best_model="eval_loss",
)
```

#### 12. SFTTrainer — Completion-Only Training
```python
DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
```
The response template `<|eot_id|><|start_header_id|>assistant<|end_header_id|>` is used to mask everything **before** the assistant turn with `-100` in the labels. This means the loss is computed only on the generated summary tokens, not on the repeated system prompt or user message. This is more sample-efficient and avoids the model overfitting prompt boilerplate.

> **Known quirk:** The Llama tokenizer requires a leading `\n` before the response template to correctly identify its token IDs. The notebook encodes `\nresponse_template` and strips the first 2 tokens to get the correct ID sequence.

#### 13. Train
```python
training_history = peft_trainer.train()
```
Progress is logged every 25 steps (loss, learning rate). Evaluation loss is also computed every 25 steps on the validation set.

#### 14. Save Adapter
```python
peft_model.save_pretrained(f"{output_dir}/best_model")
```
Only the **adapter weights** (~80–120 MB) are saved — not the full 1B-parameter base model. The saved folder contains:
- `adapter_config.json` — LoRA hyperparameters
- `adapter_model.safetensors` — trained adapter weights

---

## Notebook 2 — `Evaluate_Fine_Tuned_Model.ipynb`

### Purpose
Load the saved adapter, generate summaries with both the base model and the fine-tuned model on the test set, and compute objective evaluation metrics.

### Step-by-step walkthrough

#### 1. Install Dependencies
```bash
pip install -q -U bitsandbytes transformers peft accelerate datasets evaluate rouge_score bert_score
```
Two additional libraries compared to Notebook 1:
- `evaluate` — Hugging Face unified evaluation API
- `rouge_score` / `bert_score` — backend implementations for the metrics

#### 2. Load Base Model (same quantization config)
The same `BitsAndBytesConfig` (4-bit NF4) is used to load the base model before attaching the adapter.

#### 3. Tokenizer Differences vs. Notebook 1
```python
tokenizer.pad_token = tokenizer.eos_token   # uses EOS instead of the fine-tune pad token
```
This is intentional for inference — no training loop is involved, so the dedicated pad token is unnecessary.

#### 4. Inference Helpers

| Function | Signature | What it does |
|---|---|---|
| `generate_response` | `(model, topic, dialogue, summary='', ...)` | Formats the prompt, tokenizes, calls `model.generate()`, decodes only the new tokens |
| `get_response` | `(sample, model, tokenizer, suffix='')` | Wraps `generate_response` and stores the result as `sample['response{suffix}']` |
| `process_dataset` | `(dataset, tokenizer, model, suffix='')` | Maps `get_response` over the whole dataset |

The `suffix` argument (`_original` vs. `_peft`) allows both model outputs to live in the same HuggingFace Dataset object, making side-by-side comparison straightforward.

#### 5. Generate Baseline Outputs
```python
eval_dataset = dataset['test'].shuffle(seed=42).select(range(200))
eval_dataset = process_dataset(eval_dataset, tokenizer, model, suffix='_original')
```
200 test samples are used (out of ~1500) to keep inference time manageable.

#### 6. Load the Fine-Tuned Adapter
```python
adapter_path = './peft-dialogue-summary-training-XXXXXXXXXX/best_model'
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(model, adapter_path)
```
`PeftModel.from_pretrained` **attaches** the adapter to the already-loaded quantized base model in place. No second copy of the base model is needed.

#### 7. Generate Fine-Tuned Outputs
```python
eval_dataset = process_dataset(eval_dataset, tokenizer, model, suffix='_peft')
```
The same 200 samples are now processed again, this time with the adapter active.

#### 8. Save Results (Optional)
```python
eval_dataset.save_to_disk('eval_results')
```
Saves the dataset (including both sets of generated responses) to disk so the inference step can be skipped in future runs.

#### 9. Metric Computation

```python
def calculate_metrics(predictions, references):
    rouge  = evaluate.load('rouge')
    bert   = evaluate.load('bertscore')
    ...
```

| Metric | What it measures | Range |
|---|---|---|
| **ROUGE-1** | Unigram overlap between prediction and reference | 0 – 1 (higher = better) |
| **ROUGE-2** | Bigram overlap | 0 – 1 |
| **ROUGE-L** | Longest Common Subsequence | 0 – 1 |
| **BERTScore-P** | Semantic precision (predicted tokens matched to reference) | ~0.8 – 1.0 |
| **BERTScore-R** | Semantic recall (reference tokens matched to prediction) | ~0.8 – 1.0 |
| **BERTScore-F1** | Harmonic mean of P and R | ~0.8 – 1.0 |

ROUGE is lexical — it rewards exact word matches. BERTScore uses contextual BERT embeddings (`bert-base-uncased`) to capture **semantic similarity** even when the wording differs.

#### 10. Results Table
```python
results = pd.DataFrame([original_metrics, peft_metrics]).T
results.columns = ['Base model', 'Fine-tuned']
```
The final output is a 6-row × 2-column table showing every metric for the base model vs. the fine-tuned model side by side.

---

## Prerequisites

### Hardware
- A CUDA-capable GPU with at least **12 GB VRAM** is strongly recommended (e.g., NVIDIA T4, V100, A10).
- A free Google Colab T4 instance works for both notebooks.

### Software
- Python ≥ 3.10
- CUDA ≥ 11.8

### Access
- A Hugging Face account with the **Llama-3.2-1B-Instruct** licence accepted at [huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- A Hugging Face User Access Token added to your environment (or passed as `use_auth_token=True`).

---

## How to Run

### Step 1 — Fine-tuning
Open `Fine_Tune_Llama_3_2_1B.ipynb` and run all cells top to bottom. The adapter will be saved to a timestamped directory: `./peft-dialogue-summary-training-<unix_timestamp>/best_model/`.

### Step 2 — Evaluation
Open `Evaluate_Fine_Tuned_Model.ipynb`. Update `adapter_path` to point to the `best_model` folder produced in Step 1, then run all cells top to bottom.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| 4-bit NF4 quantization | Reduces base model VRAM from ~4 GB to ~1 GB with minimal accuracy loss |
| LoRA rank 32 with alpha 32 | Effective scale of 1 (alpha/r), common starting point; large enough to learn meaningful summarization patterns |
| Target modules: q, k, v, dense | Covers attention input and output projections — the most impactful layers for task adaptation |
| Completion-only loss masking | The model learns to summarize, not to repeat the prompt — cleaner signal |
| `paged_adamw_8bit` optimizer | Offloads optimizer states to CPU when not in use, saving ~50% GPU memory vs. full Adam |
| `load_best_model_at_end=True` | Guarantees the saved adapter corresponds to the lowest validation loss, not just the last checkpoint |
| 200 test samples for eval | Balances statistical validity with inference time on a consumer GPU |
| Both models share one base | `PeftModel.from_pretrained` attaches adapters to the already-loaded base, avoiding loading a second 1B model |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` during training | Batch size too large | Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` |
| `OSError: You are trying to access a gated repo` | Llama access not granted | Accept the licence at huggingface.co and re-authenticate |
| Adapter loads but metrics are identical to base | Wrong `adapter_path` | Verify the path points to the folder containing `adapter_config.json` |
| Very slow inference during evaluation | GPU not being used | Check `torch.cuda.is_available()` returns `True` and that model is on CUDA |
| `DataCollatorForCompletionOnlyLM` finds no responses | Response template token IDs wrong | The notebook's `[2:]` slice is Llama-specific; verify token IDs match your tokenizer version |

---

## Glossary

- **LoRA** — Low-Rank Adaptation. Injects small trainable matrices into frozen model layers.
- **QLoRA** — Quantized LoRA. Combines 4-bit quantization of the base model with LoRA adapters for full fine-tuning quality at a fraction of the memory cost.
- **NF4** — NormalFloat4. A 4-bit data type whose quantization levels are optimally spaced for normally distributed values.
- **PEFT** — Parameter-Efficient Fine-Tuning. Umbrella term for methods (LoRA, prefix tuning, etc.) that tune a small subset of parameters.
- **SFTTrainer** — Supervised Fine-Tuning Trainer from TRL. Wraps HuggingFace `Trainer` with LLM-specific defaults and completion masking.
- **ROUGE** — Recall-Oriented Understudy for Gisting Evaluation. Lexical overlap metric family for summarization.
- **BERTScore** — Evaluation metric that computes token-level cosine similarity using BERT embeddings.
- **DXA** — Abbreviation for "device-independent pixels" used in OOXML. Unrelated to this project; ignore.

## High-Performance Inference with vLLM

This project can also leverage **vLLM** for faster and more memory-efficient inference compared to standard Hugging Face generation.

vLLM uses **PagedAttention** and optimized GPU scheduling to significantly improve throughput and latency, especially for batch inference or production workloads.

---

### Installation

```bash
pip install vllm
```

> Requires a CUDA-enabled GPU. Best supported on Linux or WSL.

---

### Load Model with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="float16"
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)
```

---

### Prompt Template

```python
def build_prompt(topic, dialogue):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert on summarizing conversations considering a particular topic.
Answer with the summary only.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Topic: {topic}
Conversation: {dialogue}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
```

---

### Generate Output

```python
prompt = build_prompt(sample["topic"], sample["dialogue"])

outputs = llm.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text.strip()

print(response)
```

---

### ️ Using LoRA Adapter with vLLM

vLLM does not directly support PEFT LoRA adapters, so you must merge the adapter into the base model before using it.

#### Merge LoRA Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype="float16"
)

model = PeftModel.from_pretrained(base_model, "path_to_adapter")

# Merge LoRA weights into base model
model = model.merge_and_unload()

model.save_pretrained("merged_model")
```

#### Load Merged Model in vLLM

```python
from vllm import LLM

llm = LLM(model="merged_model")
```

---

### Optional: OpenAI-Compatible API Server

Run:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model merged_model \
    --port 8000
```

Then query:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="merged_model",
    messages=[{"role": "user", "content": "Summarize this dialogue..."}],
)

print(response.choices[0].message.content)
```

---

### When to Use vLLM

| Scenario | Recommendation |
|---|---|
| Training / Fine-tuning | Transformers + PEFT |
| Small-scale evaluation | Transformers |
| Fast inference / batch processing | vLLM |
| Production deployment | vLLM |

---

### Summary

- vLLM enables fast, scalable inference
- Merge LoRA adapters before using vLLM
- Ideal for serving and large-scale generation
