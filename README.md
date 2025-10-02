

## Data Descriptions

SwissProtCLAP

[SwissProtCLAP](https://huggingface.co/datasets/chao1224/ProteinDT/tree/main)


In the data preprocessing folder, there are **four Colab notebooks** that form an end-to-end pipeline from dataset checks to CoT generation, structure tokenization (Foldseek 3Di), and a forward/loss smoke test for understanding model.

**Run them in this order:**
1. **`ProteinDT_molreasoner_pro2dec_demo_colab.ipynb`** — Demo / check all APIs and dataset
2. **`ProteinDT_molreasoner_pro2dec_cot_colab.ipynb`** — Generate Chain-of-Thought (CoT)
3. **`ProteinDT_molreasoner_pro2dec_PDB3Di_colab.ipynb`** — AlphaFold PDB → Foldseek 3Di
4. **`BigProteinQwen_Colab_Debugging.ipynb`** — Test each model component; forward & loss

## Quick Start

1. Open each notebook in **Google Colab**.
2. Set the paths at the top (e.g., `BASE_DIR`) and required creds (e.g., `OPENAI_API_KEY`).
3. Run the notebooks **in order** (1→4). Notebooks are resume-friendly and avoid redoing work where possible.

---

## Notebooks

### 1) `ProteinDT_molreasoner_pro2dec_demo_colab.ipynb` — Demo / data & API checks
**Purpose**
- Sanity-check the **ProteinDT** dataset (e.g., SwissProtCLAP `protein_sequence.txt` ↔ `text_sequence.txt` alignment, counts).
- Verify external structure APIs:
  - **AlphaFold** (view/download a sample structure)
  - **RCSB/PDBe** (map UniProt → experimental PDB IDs)
- Confirm Google Drive I/O and basic utilities.

**Inputs**
- Local copy of ProteinDT under `BASE_DIR`.

**Outputs**
- Printed summaries, small CSVs, and optional sample downloads under:
  - `BASE_DIR/`
  - `BASE_DIR/downloads/<UniProt>/`

---

### 2) `ProteinDT_molreasoner_pro2dec_cot_colab.ipynb` — Generate CoT
**Purpose**
- Build prompts from ProteinDT (sequence → description).
- Submit **OpenAI Batch** jobs (chunked, resume-safe) and **fetch** results.
- Produce two SFT files:
  - **Parsed CoT**: `<thinking>…</thinking>\n\n<answer>…</answer>`
  - **Raw**: unmodified model output for audit.

**Key Features**
- Loose parsing (tolerates missing closing tags or answer-only).
- Optional **fixed-answer** mode: provide the ground-truth `<answer>`, model fills only `<thinking>`.

**Outputs** (under something like)
- `BASE_DIR/gpt_batch_protein2desc*/`
  - `*_input.jsonl`, `*_meta.json`, `*_batch_info.json`
  - `protein2desc_cot_sft.json` (parsed CoT)
  - `protein2desc_cot_raw.json` (raw text)

---

### 3) `ProteinDT_molreasoner_pro2dec_PDB3Di_colab.ipynb` — AlphaFold PDB → Foldseek 3Di
**Purpose**
- For your UniProt set (e.g., from the CoT batches), ensure **AlphaFold PDB** exists locally (v4→v3→v2; **PDB only**).
- Convert structures to **Foldseek 3Di tokens** for **all chains** found (A–Z, 0–9).
- Concatenate chains with special tokens:
  - chain separator: `<|chain_sep|>`
  - optional per-chain tag: `<|chain:{ID}|>`

**Outputs** (under)
- `BASE_DIR/sft_build/foldseek_3di_pdb/`
  - `<UID>.3di.txt` — concatenated 3Di
  - `<UID>.aa.txt` — concatenated AA (aligned with 3Di)
  - `<UID>.chains.json` — chain metadata (IDs, lengths, file used)
  - `3di_manifest_pdb.csv` — manifest
  - `3di_failed_pdb.txt` — failures (if any)

---

### 4) `BigProteinQwen_Colab_Debugging.ipynb` — Model components & forward/loss
**Purpose**
- Unit-style checks for model components (tokenizers/encoders).
- Data collation from your SFT (CoT + 3Di).
- **Forward pass** and **loss** computation on a mini-batch (smoke test).

**Outputs**
- Printed shapes, losses, and sanity diagnostics. Optional logs under your `BASE_DIR`.

---

## Requirements

- **Colab** (recommended) or Python 3.10+.
- Common packages: `pandas`, `tqdm`, `requests`, `torch`, `openai`.
- **Foldseek** binary for 3Di:
  - Place at `bin/foldseek` (or update the path in notebook)
  - Make it executable:
    ```bash
    chmod +x bin/foldseek
    ```
- **OpenAI API key** (for CoT):
  ```python
  import os
  os.environ["OPENAI_API_KEY"] = "sk-..."





This repo provides three entrypoints:

- **`train_prefix_qwen.py`** — single‑GPU trainer (gradient accumulation; optional encoder finetune).  
- **`train_prefix_qwen_fsdp.py`** — multi‑GPU trainer using 🤗 **Accelerate** (FSDP/ZeRO), **slot‑based ProTrek `.pt` loading**, uniform 2048‑D protein vectors, explicit dtype control.  
- **`train_prefix_qwen_fsdp_offload1.py`** — FSDP full‑shard with **CPU offload** (plus optional 8‑bit/Adafactor optimizers).

---

## How it works (high‑level)
- Each example contains a **prompt**, **response**, and optional **protein inputs**:
  - `aa_seq` — amino‑acid sequence (string) → **1024‑D** embedding
  - `stru_str` — Foldseek 3Di structural string (string) → **1024‑D** embedding
- The two 1024‑D embeddings are **concatenated** to **2048‑D**; if one modality is missing, it is **zero‑padded**.
- A small **projector MLP** maps 2048‑D → `prefix_len × hidden_size` and yields a **soft prefix** (one or more learned tokens) directly in the LLM’s hidden space.
- At training time, the prefix is **prepended** to the token embeddings of the textual prompt.
- **Loss is masked** over the prefix and prompt (and its EOS) so cross‑entropy supervises **only the response tokens**.

---

## File layout
```
protein_encoder.py                 # sequence encoder wrapper
structure_encoder.py               # 3Di structure encoder wrapper
train_prefix_qwen.py               # single‑GPU trainer
train_prefix_qwen_fsdp.py          # multi‑GPU trainer (FSDP/ZeRO)
train_prefix_qwen_fsdp_offload1.py # multi‑GPU trainer (FSDP + CPU offload)
```

Both multi‑GPU scripts load encoder weights from a **ProTrek** `.pt` checkpoint using **slot IDs** (e.g., `prot_slot=1`, `stru_slot=3`).

---

## Data format (JSONL)
Each line is a JSON object:
```json
{
  "prompt": "Describe the likely function of this protein.",
  "response": "This appears to be an enzyme with possible hydrolase activity.",
  "aa_seq": "MGDVEK...",         // optional
  "stru_str": "ACDEFGH..."       // optional (Foldseek 3Di string)
}
```
The collator builds `[prompt + EOS] + [response + EOS]` and masks labels to supervise **only** the response (EOS included).

---

## Encoder initialization (ProTrek `.pt` + slots)
- **Architectures** come from `--protein-config` and `--structure-config` (ESM‑style configs or local dirs).
- **Weights** come from a ProTrek **`.pt`** checkpoint passed via `--protrek-ckpt`. The state dict is split by numeric **slot IDs**; `--prot-slot` and `--stru-slot` choose which sub‑model to load for **protein** and **structure** encoders respectively.
- If a slot is missing, the script prints a warning and keeps random init for that encoder.

---

## Dtype
- **Single‑GPU (`train_prefix_qwen.py`)**: `--dtype {fp32,bf16,fp16}` controls LLM load precision. CE is computed stably; projectors/encoders are aligned appropriately.
- **FSDP (`train_prefix_qwen_fsdp.py`)**: `--dtype {auto,fp32,fp16,bf16}` controls the **load dtype**. Projectors are instantiated in the **same dtype** as the LLM. Your Accelerate config may also specify mixed precision for autocast.
- **FSDP + Offload (`train_prefix_qwen_fsdp_offload1.py`)**: precision managed by the Accelerate config; offload variant keeps CE stable and supports memory‑saving optimizer options.

---

## Rank‑sharded streaming
The FSDP trainers stream JSONL and **shard lines by process rank** (`RANK/WORLD_SIZE`), so each GPU reads a disjoint subset without duplication.

---

## Quickstart

### A) Single‑GPU
```bash
python train_prefix_qwen.py \
  --train-file /data/train.jsonl \
  --val-file   /data/val.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --protein-config   facebook/esm2_t12_35M_UR50D \
  --structure-config facebook/esm2_t12_35M_UR50D \
  --protrek-ckpt     /weights/ProTrek_35M.pt \
  --prot-slot 1 --stru-slot 3 \
  --dtype fp32 \
  --prefix-len 4 \
  --batch-size 4 --accum-steps 4 --max-len 1024 \
  --epochs 1 --lr 1e-5 \
  --save-dir ./runs --save-every 1000
```

### B) Multi‑GPU (FSDP/ZeRO)
Create an Accelerate YAML (example below), then:
```bash
accelerate launch --config_file accelerate_fsdp_bf16.yaml train_prefix_qwen_fsdp.py \
  --train-file /data/train.jsonl \
  --val-file   /data/val.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --protein-config   facebook/esm2_t12_35M_UR50D \
  --structure-config facebook/esm2_t12_35M_UR50D \
  --protrek-ckpt     /weights/ProTrek_35M.pt \
  --prot-slot 1 --stru-slot 3 \
  --dtype bf16 \
  --prefix-len 4 \
  --batch-size 1 --accum-steps 8 --max-len 2048 \
  --epochs 1 --lr 1e-5 \
  --save-dir ./runs_fsdp --save-every 1000
```

**`accelerate_fsdp_bf16.yaml` (example):**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 4             # number of GPUs
mixed_precision: bf16
gradient_accumulation_steps: 8
downcast_bf16: 'no'
fsdp_config:
  sharding_strategy: FULL_SHARD
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen2DecoderLayer,LlamaDecoderLayer,GPTNeoXLayer
  backward_prefetch: BACKWARD_PRE
  state_dict_type: FULL_STATE_DICT
  sync_module_states: true
  cpu_offload: false
  use_orig_params: true
  limit_all_gathers: true
```

### C) FSDP + **CPU offload**
```bash
accelerate launch --config_file accelerate_cpu_offload_bf16.yaml train_prefix_qwen_fsdp_offload1.py \
  --train-file /data/train.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --protein-config   facebook/esm2_t12_35M_UR50D \
  --structure-config facebook/esm2_t12_35M_UR50D \
  --protrek-ckpt     /weights/ProTrek_35M.pt \
  --prot-slot 1 --stru-slot 3 \
  --prefix-len 4 --batch-size 1 --accum-steps 8 \
  --max-len 1024 --epochs 1 --lr 1e-4 \
  --optimizer adam8bit --train-encoders \
  --save-dir ./runs_offload
```
*This variant performs selected computations/parameters on CPU to fit larger models; expect slower throughput.*

---

## Common arguments
- `--train-file`, `--val-file` — JSONL paths.
- `--model-name` — Hugging Face model ID or local path (e.g., `Qwen/Qwen2.5-7B-Instruct`).
- `--protein-config`, `--structure-config` — encoder architectures (ESM‑style or local).
- `--protrek-ckpt`, `--prot-slot`, `--stru-slot` — load encoder weights from ProTrek `.pt` via slots.
- `--prefix-len` — number of soft prefix tokens (`--single-token-prefix` in FSDP trainer for 1‑token mode).
- `--batch-size`, `--accum-steps`, `--max-len`, `--epochs`, `--lr`, `--warmup-ratio`, `--weight-decay`.
- `--save-dir`, `--save-every`, `--log-every`, `--seed`.
- `--dtype` — load dtype; in FSDP trainer also determines projector dtype.

---

## Troubleshooting
- **Slot not found**: verify `--prot-slot` / `--stru-slot` match the ckpt layout; otherwise encoder stays random‑init (warning printed).
- **OOM**: for large LLMs use FSDP full‑shard with `bf16`, increase `--accum-steps`, reduce `--max-len`, or switch to the CPU‑offload trainer and/or 8‑bit optimizer.
- **dtype mismatch**: keep `--dtype` consistent with Accelerate’s `mixed_precision`. The FSDP trainer instantiates the projector in the LLM’s dtype to avoid matmul errors.
- **No supervised tokens**: ensure responses are non‑empty; the prompt (and its EOS) is fully masked in labels.

---

starting command: based on python 3.10

!nvidia-smi
%pip -q install --upgrade pip
%pip install -q --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
%pip -q install transformers==4.56.1 huggingface_hub==0.35.0 tqdm safetensors
