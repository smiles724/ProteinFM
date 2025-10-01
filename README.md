

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
