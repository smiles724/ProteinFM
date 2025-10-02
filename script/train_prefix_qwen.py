#!/usr/bin/env python3
"""
Protein-conditioned SFT with a soft prefix on top of a Causal LM (e.g., Qwen2.5).
- Streams JSONL examples (prompt/response + optional aa_seq / stru_str)
- Encodes protein sequence/structure with ESM encoders
- Projects [prot|stru] -> PREFIX_LEN soft tokens prepended to the prompt
- Masks loss over prefix + prompt; supervises only the response tokens
- Supports fp32 / bf16 / fp16 with a stable fp32 loss path
- Single-GPU training loop with gradient accumulation & periodic checkpointing

Expected JSONL schema (one JSON per line):
{
  "prompt": "Describe the ...",
  "response": "It ...",
  "aa_seq": "MGDVEK...",
  "stru_str": "acdefgh..."   # Foldseek 3Di string, lowercase. Optional.
}

Usage example:
python train_prefix_qwen.py   --train-file /data/train.jsonl   --val-file /data/val.jsonl   --model-name Qwen/Qwen2.5-0.5B-Instruct   --protein-config /weights/esm2_t12_35M_UR50D   --structure-config /weights/foldseek_t12_35M   --protrek-ckpt /weights/ProTrek_35M.pt   --prot-slot 1 --stru-slot 3   --dtype fp32   --batch-size 8 --accum-steps 4 --max-len 1024 --prefix-len 4   --epochs 1 --lr 1e-5   --freeze-llm false   --save-dir ./ckpts --save-every 1000 --eval-every 2000
"""

import os, json, math, argparse, time, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local modules: keep these python files in the same folder
import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod

# ------------------------
# Utilities
# ------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

# ------------------------
# JSONL streaming dataset
# ------------------------

class JsonlStream(IterableDataset):
    """
    Streams a JSONL file line-by-line to avoid holding the full dataset in memory.
    Yields dicts with keys: prompt, response, aa_seq (opt), stru_str (opt).
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                # Minimal schema sanity; you can extend as needed
                if "prompt" not in ex or "response" not in ex:
                    continue
                yield {
                    "prompt": ex["prompt"],
                    "response": ex["response"],
                    "aa_seq": ex.get("aa_seq", None),
                    "stru_str": ex.get("stru_str", None),
                }

# ------------------------
# Collator that builds batched text ids and protein vectors
# ------------------------

@dataclass
class CollateConfig:
    tokenizer: Any
    max_len: int
    device: torch.device

class PadOnlyTextCollator:
    """
    - tokenizes prompt/response separately (mask prompts in labels)
    - pads to the max length in the batch
    - DOES NOT compute protein vectors (so we can backprop through encoders)
    Returns:
      input_ids, attention_mask, labels, plus the raw aa_seq / stru_str lists
    """
    def __init__(self, cfg: CollateConfig):
        self.tok = cfg.tokenizer
        self.max_len = cfg.max_len
        self.device = cfg.device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts   = [b["prompt"]   for b in batch]
        responses = [b["response"] for b in batch]

        enc_p = self.tok(prompts, add_special_tokens=False)
        enc_r = self.tok([r + self.tok.eos_token for r in responses], add_special_tokens=False)

        ids_list, prompt_lens = [], []
        T_max = 0
        for i in range(len(batch)):
            ids_p = enc_p["input_ids"][i]
            ids_r = enc_r["input_ids"][i]
            ids   = (ids_p + ids_r)[: self.max_len]
            ids_list.append(ids)
            p_keep = min(len(ids_p), len(ids))  # prompt portion to mask in labels
            prompt_lens.append(p_keep)
            T_max = max(T_max, len(ids))

        pad_id = self.tok.pad_token_id
        input_ids = torch.full((len(batch), T_max), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), T_max, dtype=torch.long)
        labels    = torch.full((len(batch), T_max), -100, dtype=torch.long)

        for i, ids in enumerate(ids_list):
            t = len(ids)
            input_ids[i, :t] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :t] = 1
            L = [-100]*prompt_lens[i] + ids[prompt_lens[i]:]
            labels[i, :t] = torch.tensor(L, dtype=torch.long)

        # pass raw sequences so we can run encoders with/without grad in forward
        aa_list   = [b.get("aa_seq")   for b in batch]
        stru_list = [b.get("stru_str") for b in batch]

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
            "aa_seq":         aa_list,
            "stru_str":       stru_list,
        }

# ------------------------
# Encoder loading (ProTrek slots)
# ------------------------

def load_encoders_and_ckpt(protein_config: str,
                           structure_config: str,
                           ckpt_path: Optional[str],
                           prot_slot: int,
                           stru_slot: int,
                           device: torch.device):
    ProteinEncoder   = protein_encoder_mod.ProteinEncoder
    StructureEncoder = structure_encoder_mod.StructureEncoder

    prot_enc = ProteinEncoder(protein_config, out_dim=1024, load_pretrained=False).to(device).eval()
    stru_enc = StructureEncoder(structure_config, out_dim=1024, load_pretrained=False).to(device).eval()

    if ckpt_path and os.path.exists(ckpt_path):
        sd_raw = torch.load(ckpt_path, map_location="cpu")
        sd = sd_raw.get("model", sd_raw.get("state_dict", sd_raw))
        slots = {}
        for k, v in sd.items():
            head = k.split(".", 1)[0]
            if head.isdigit():
                slots.setdefault(int(head), {})[k[len(head)+1:]] = v

        def drop_extras(sub: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {k: v for k, v in sub.items() if "embeddings.position_ids" not in k}

        if prot_slot in slots:
            mp, up = prot_enc.load_state_dict(drop_extras(slots[prot_slot]), strict=False)
            print(f"[ProteinEncoder] loaded from slot {prot_slot} | missing={len(mp)} unexpected={len(up)}")
        else:
            print(f"[ProteinEncoder] WARNING: slot {prot_slot} not found; skipping ckpt load.")

        if stru_slot in slots:
            ms, us = stru_enc.load_state_dict(drop_extras(slots[stru_slot]), strict=False)
            print(f"[StructureEncoder] loaded from slot {stru_slot} | missing={len(ms)} unexpected={len(us)}")
        else:
            print(f"[StructureEncoder] WARNING: slot {stru_slot} not found; skipping ckpt load.")
    else:
        print("No ProTrek checkpoint provided or path not found; encoders stay random-init.")

    # By default we keep encoders in eval() and do not backprop through them for throughput.
    # If you want to train encoders, call .train() and include their params in the optimizer.
    return prot_enc, stru_enc

# ------------------------
# Training
# ------------------------

def stable_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy computed in float32 for numerical stability.
    Shapes:
      logits: (B, L, V)
      labels: (B, L) with -100 masked labels
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load tokenizer & LLM ----
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    model_dtype = dtype_map[args.dtype]

    llm = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=model_dtype).to(device)
    we = llm.get_input_embeddings()
    hidden_size = llm.config.hidden_size
    print(f"Loaded LLM: {args.model_name} | hidden_size={hidden_size} | dtype={model_dtype}")

    # ---- Load encoders & optional ProTrek ckpt ----
    prot_enc, stru_enc = load_encoders_and_ckpt(
        args.protein_config, args.structure_config, args.protrek_ckpt, args.prot_slot, args.stru_slot, device
    )
    if args.train_encoders:
        prot_enc.train(); stru_enc.train()
    else:
        prot_enc.eval(); stru_enc.eval()
        for p in prot_enc.parameters(): p.requires_grad = False
        for p in stru_enc.parameters(): p.requires_grad = False

    # ---- Projector & optional gate ----
    projector = nn.Sequential(
        nn.Linear(1024+1024, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size * args.prefix_len),
    ).to(device, dtype=model_dtype)

    prefix_gate = None
    if args.prefix_gate is not None:
        g = torch.tensor(float(args.prefix_gate), device=device, dtype=model_dtype)
        if args.learnable_gate:
            prefix_gate = nn.Parameter(g)
        else:
            prefix_gate = g  # constant tensor
    # params to optimize
    optim_params = list(projector.parameters())
    if prefix_gate is not None and isinstance(prefix_gate, nn.Parameter):
        optim_params += [prefix_gate]
    if not args.freeze_llm:
        optim_params += list(llm.parameters())
    if args.train_encoders:
        optim_params += list(prot_enc.parameters()) + list(stru_enc.parameters())

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Data ----
    train_ds = JsonlStream(args.train_file)
    val_ds   = JsonlStream(args.val_file) if args.val_file else None

    collate = PadOnlyTextCollator(CollateConfig(
        tokenizer=tok, max_len=args.max_len, device=device
    ))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=collate)
    val_loader   = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=False, collate_fn=collate)
    # ---- Train loop ----
    llm.train(not args.freeze_llm); projector.train()
    if args.train_encoders:
        prot_enc.train()
        stru_enc.train()
    else:
        prot_enc.eval()
        stru_enc.eval()
    global_step = 0
    t0 = time.time()

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=model_dtype) 
                    if (device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)) else None)
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and model_dtype == torch.float16) else None

    def forward_batch(batch) -> Tuple[torch.Tensor, int]:
        input_ids      = batch["input_ids"].to(device)
        attn_mask_text = batch["attention_mask"].to(device)
        labels_text    = batch["labels"].to(device)
        aa_list        = batch["aa_seq"]
        stru_list      = batch["stru_str"]

        B, T = input_ids.shape
        text_embeds = we(input_ids).to(model_dtype)  # (B, T, H)

        # --- run encoders here so they get gradients when train_encoders=True ---
        def zeros_vec(n):  # CPU float32; cast later
            return torch.zeros(n, 1024, dtype=torch.float32)

        need_prot = any(bool(s) for s in aa_list)
        need_stru = any(bool(s) for s in stru_list)

        if args.train_encoders:
            # compute with grad
            prot_vec_list = []
            if need_prot:
                seqs = [s for s in aa_list if s]
                pv = prot_enc.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)  # (n,1024) float32
                prot_vec_list = [pv[i] for i in range(pv.shape[0])]
            stru_vec_list = []
            if need_stru:
                seqs = [s for s in stru_list if s]
                sv = stru_enc.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)   # (n,1024) float32
                stru_vec_list = [sv[i] for i in range(sv.shape[0])]
        else:
            # speed path (no grad into encoders)
            with torch.no_grad():
                prot_vec_list = []
                if need_prot:
                    seqs = [s for s in aa_list if s]
                    pv = prot_enc.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)
                    prot_vec_list = [pv[i] for i in range(pv.shape[0])]
                stru_vec_list = []
                if need_stru:
                    seqs = [s for s in stru_list if s]
                    sv = stru_enc.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)
                    stru_vec_list = [sv[i] for i in range(sv.shape[0])]

        # stitch back to B x 1024 per-modality in original order
        prot_full = zeros_vec(B)
        stru_full = zeros_vec(B)
        ip = 0; is_ = 0
        for i in range(B):
            if aa_list[i]:
                prot_full[i] = prot_vec_list[ip]; ip += 1
            if stru_list[i]:
                stru_full[i] = stru_vec_list[is_]; is_ += 1

        pvec = torch.cat([prot_full, stru_full], dim=1).to(device).to(model_dtype)  # (B, 2048)

        # --- prefix + text concatenation ---
        pref = projector(pvec).view(B, args.prefix_len, hidden_size)
        if prefix_gate is not None:
            pref = pref * (prefix_gate if isinstance(prefix_gate, torch.Tensor)
                        else torch.tensor(prefix_gate, device=device, dtype=model_dtype))

        inputs_embeds = torch.cat([pref, text_embeds], dim=1)  # (B, P+T, H)
        attn_mask     = torch.cat([torch.ones(B, args.prefix_len, device=device, dtype=torch.long),
                                attn_mask_text], dim=1)
        labels        = torch.cat([torch.full((B, args.prefix_len), -100, device=device, dtype=torch.long),
                                labels_text], dim=1)

        # forward + stable fp32 CE
        if autocast_ctx is None:
            out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
            loss = stable_ce_loss(out.logits, labels)
        else:
            with autocast_ctx:
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
            loss = stable_ce_loss(out.logits, labels)

        n_valid = int((labels != -100).sum().item())
        return loss, n_valid

    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        tokens_supervised = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            loss, n_valid = forward_batch(batch)
            if n_valid == 0 or not torch.isfinite(loss):
                print("Skipping batch: no supervised tokens or non-finite loss.")
                continue

            if scaler is not None:
                scaler.scale(loss / args.accum_steps).backward()
            else:
                (loss / args.accum_steps).backward()

            running += float(loss.detach().cpu())
            tokens_supervised += n_valid

            if step % args.accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    avg = running / 50.0
                    print(f"[ep {epoch}] step {global_step} | loss={avg:.4f} | supervised_tokens={tokens_supervised} | time={human_time(time.time()-t0)}")
                    running = 0.0; tokens_supervised = 0

                if args.save_every and (global_step % args.save_every == 0):
                    os.makedirs(args.save_dir, exist_ok=True)
                    path = os.path.join(args.save_dir, f"ckpt_step{global_step}.pt")
                    save_state = {
                        "projector": projector.state_dict(),
                        "llm": (llm.state_dict() if not args.freeze_llm else None),
                        "encoders": {
                            "protein":  prot_enc.state_dict() if args.train_encoders else None,
                            "structure": stru_enc.state_dict() if args.train_encoders else None,
                        },
                        "prefix_gate": (prefix_gate.data if isinstance(prefix_gate, nn.Parameter) else prefix_gate),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    torch.save(save_state, path)
                    print(f"Saved checkpoint to {path}")

                if args.eval_every and val_loader and (global_step % args.eval_every == 0):
                    llm.eval(); projector.eval()
                    if args.train_encoders:
                        prot_enc.eval(); stru_enc.eval()
                    with torch.no_grad():
                        vl, vc = 0.0, 0
                        for vb in val_loader:
                            vloss, vn = forward_batch(vb)
                            if vn == 0 or not torch.isfinite(vloss):
                                continue
                            vl += float(vloss.detach().cpu()); vc += 1
                        if vc > 0:
                            vavg = vl / vc
                            print(f"  ↳ val loss @ step {global_step}: {vavg:.4f}")
                            best_val = min(best_val, vavg)
                    llm.train(not args.freeze_llm); projector.train()
                    if args.train_encoders:
                        prot_enc.train(); stru_enc.train()

        # end epoch
        print(f"Finished epoch {epoch}. Elapsed {human_time(time.time()-t0)}")

    # Final save
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, "final.pt")
        save_state = {
            "projector": projector.state_dict(),
            "llm": (llm.state_dict() if not args.freeze_llm else None),
            "encoders": {
                "protein":  prot_enc.state_dict() if args.train_encoders else None,
                "structure": stru_enc.state_dict() if args.train_encoders else None,
            },
            "prefix_gate": (prefix_gate.data if isinstance(prefix_gate, nn.Parameter) else prefix_gate),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "final_step": global_step,
        }
        torch.save(save_state, path)
        print(f"Saved final checkpoint to {path}")

# ------------------------
# CLI
# ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Protein-conditioned SFT with soft prefix")
    # Data
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--val-file",   type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-len", type=int, default=1024)
    # Model
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", type=str, choices=["fp32","bf16","fp16"], default="fp32")
    p.add_argument("--prefix-len", type=int, default=1)
    p.add_argument("--prefix-gate", type=float, default=0.1, help="scale on prefix embeddings; set None to disable")
    p.add_argument("--learnable-gate", action="store_true", help="make prefix gate a learnable parameter")
    p.add_argument("--freeze-llm", action="store_true")
    p.add_argument("--train-encoders", action="store_true", help="set to backprop through encoders")
    # Encoders
    p.add_argument("--protein-config", type=str, required=True)
    p.add_argument("--structure-config", type=str, required=True)
    p.add_argument("--protrek-ckpt", type=str, default=None)
    p.add_argument("--prot-slot", type=int, default=1)
    p.add_argument("--stru-slot", type=int, default=3)
    # Optim
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    # Save/eval
    p.add_argument("--save-dir", type=str, default="./runs")
    p.add_argument("--save-every", type=int, default=0, help="save every N optimizer steps (0=disabled)")
    p.add_argument("--eval-every", type=int, default=0, help="eval every N optimizer steps (0=disabled)")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
