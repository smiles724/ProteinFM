#!/usr/bin/env python3
"""
Multi-GPU protein-conditioned SFT with a learned prefix for Qwen 2.5.
- Uses 🤗 Accelerate (FSDP/ZeRO) so models larger than one GPU can train.
- Loads encoders from a ProTrek .pt checkpoint by slot (e.g., prot_slot=1, stru_slot=3).
- Always encodes to a uniform 2048-D vector (seq 1024 || stru 1024), zero-padding missing modality.
- Projectors are dtype-safe and output directly to LLM hidden size.
- Sharded JSONL streaming: each rank reads a disjoint subset of lines.
"""

import os, json, math, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod

# ------------------------
# dtype helpers
# ------------------------

DTYPE_MAP = {
    "fp32": torch.float32, "float32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "auto": None, "default": None, None: None
}

def resolve_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = str(dtype_str).lower() if dtype_str is not None else None
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype_str}. Use one of: fp32, fp16, bf16, auto.")
    return DTYPE_MAP[key]

def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

# ------------------------
# JSONL streaming dataset (rank-sharded)
# ------------------------

class JsonlStream(IterableDataset):
    """
    Streams a JSONL file; each rank reads a modulo shard.
    Expects per-line objects with: prompt, response, (optional) aa_seq, stru_str.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        rank = int(os.environ.get("RANK", "0"))
        world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if (i % world) != rank:
                    continue
                ex = json.loads(line)
                if "prompt" not in ex or "response" not in ex:
                    continue
                yield {
                    "prompt": ex["prompt"],
                    "response": ex["response"],
                    "aa_seq": ex.get("aa_seq", None),
                    "stru_str": ex.get("stru_str", None),
                }

# ------------------------
# Collator: tokenizes prompt/response and masks prompt+EOS in labels
# ------------------------

@dataclass
class CollateCfg:
    tokenizer: Any
    max_len: int

class PadAndMaskCollator:
    """
    - Tokenizes prompt and response separately to know the boundary for masking.
    - Pads to max length in batch.
    - Returns: input_ids, attention_mask, labels, aa_seq list, stru_str list
    """
    def __init__(self, cfg: CollateCfg):
        self.tok = cfg.tokenizer
        self.max_len = cfg.max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts  = [b["prompt"]  for b in batch]
        replies  = [b["response"] for b in batch]
        aa_list  = [b.get("aa_seq") for b in batch]
        stru_list= [b.get("stru_str") for b in batch]

        t_prompt = self.tok(prompts, add_special_tokens=False)
        t_reply  = self.tok(replies, add_special_tokens=False)

        input_ids = []
        labels    = []
        for p_ids, r_ids in zip(t_prompt["input_ids"], t_reply["input_ids"]):
            ids = p_ids + [self.tok.eos_token_id] + r_ids + [self.tok.eos_token_id]
            Lp = len(p_ids) + 1  # prompt + EOS
            lab = [-100]*Lp + r_ids + [self.tok.eos_token_id]
            input_ids.append(ids[:self.max_len])
            labels.append(lab[:self.max_len])

        enc = self.tok.pad(
            {"input_ids": input_ids},
            padding=True, max_length=self.max_len, return_tensors="pt"
        )
        attn = enc.attention_mask
        ids  = enc.input_ids

        # Pad labels to same shape
        maxT = ids.shape[1]
        padded_labels = torch.full((len(labels), maxT), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            L = min(len(lab), maxT)
            padded_labels[i, :L] = torch.tensor(lab[:L], dtype=torch.long)

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": padded_labels,
            "aa_seq": aa_list,
            "stru_str": stru_list,
        }

# ------------------------
# Prefix projector (in_dim -> mid -> out_hidden*out_tokens), dtype-safe
# ------------------------

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_hidden: int, out_tokens: int, dropout: float = 0.1, dtype=None, device=None):
        super().__init__()
        self.out_tokens = out_tokens
        self.out_hidden = out_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_hidden * out_tokens, bias=True),
        )
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, protein_vec: torch.Tensor) -> torch.Tensor:
        # protein_vec: [B, D]
        x = self.net(protein_vec)  # [B, T*H]
        B = x.size(0)
        T = self.out_tokens
        H = self.out_hidden
        return x.view(B, T, H)

# ------------------------
# Big model
# ------------------------

class BigProteinQwen(nn.Module):
    def __init__(
        self,
        model_name: str,
        protein_config: str,
        structure_config: str,
        protrek_ckpt: str = None,
        prot_slot: int = 1,
        stru_slot: int = 3,
        single_token_prefix: bool = False,
        prefix_len: int = 4,
        proj_hid: int = 1024,
        dropout: float = 0.1,
        train_encoders: bool = False,
        dtype_str: str = "auto",
    ):
        super().__init__()
        load_dtype = resolve_dtype(dtype_str)
        if load_dtype is None:
            load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.llm.config.hidden_size
        self.prefix_len = 1 if single_token_prefix else prefix_len

        # Encoders (arch from configs; weights from ProTrek slots below)
        self.protein_encoder   = protein_encoder_mod.ProteinEncoder(protein_config, out_dim=1024, load_pretrained=False)
        self.structure_encoder = structure_encoder_mod.StructureEncoder(structure_config, out_dim=1024, load_pretrained=False)

        # ---- ProTrek slot-based loading ----
        if protrek_ckpt and os.path.exists(protrek_ckpt):
            sd_raw = torch.load(protrek_ckpt, map_location="cpu")
            sd = sd_raw.get("model", sd_raw.get("state_dict", sd_raw))
            slots = {}
            for k, v in sd.items():
                head = k.split(".", 1)[0]
                if head.isdigit():
                    slots.setdefault(int(head), {})[k[len(head)+1:]] = v

            def drop_extras(sub):
                return {k: v for k, v in sub.items() if "embeddings.position_ids" not in k}

            if prot_slot in slots:
                mp, up = self.protein_encoder.load_state_dict(drop_extras(slots[prot_slot]), strict=False)
                print(f"[ProteinEncoder] loaded from slot {prot_slot} | missing={len(mp)} unexpected={len(up)}")
            else:
                print(f"[ProteinEncoder] WARNING: slot {prot_slot} not found; skipping ckpt load.")

            if stru_slot in slots:
                ms, us = self.structure_encoder.load_state_dict(drop_extras(slots[stru_slot]), strict=False)
                print(f"[StructureEncoder] loaded from slot {stru_slot} | missing={len(ms)} unexpected={len(us)}")
            else:
                print(f"[StructureEncoder] WARNING: slot {stru_slot} not found; skipping ckpt load.")
        else:
            print("No ProTrek checkpoint provided or path not found; encoders stay random-init.")

        # projector (outputs directly to LLM hidden size, dtype-aligned to LLM)
        model_dtype = next(self.llm.parameters()).dtype
        self.proj_1024 = PrefixProjector(1024, proj_hid, self.hidden_size, self.prefix_len, dropout, dtype=model_dtype)
        self.proj_2048 = PrefixProjector(2048, proj_hid, self.hidden_size, self.prefix_len, dropout, dtype=model_dtype)

        # Freeze encoders if requested
        if not train_encoders:
            for p in self.protein_encoder.parameters():   p.requires_grad = False
            for p in self.structure_encoder.parameters(): p.requires_grad = False

    # -------- protein encoding to a uniform 2048-D vector ----------
    def encode_protein_batch(self, aa_list: List[Optional[str]], stru_list: List[Optional[str]]) -> torch.Tensor:
        """
        Encode a batch of (aa_seq, stru_str) into a uniform 2048-D vector per example:
        - seq encoder -> 1024, stru encoder -> 1024
        - missing modality is padded with zeros
        - result shape: [B, 2048], dtype/device aligned to LLM
        """
        B = len(aa_list)
        device = next(self.llm.parameters()).device
        dtype = next(self.llm.parameters()).dtype

        # Preallocate zeros
        seq_out  = torch.zeros(B, 1024, device=device, dtype=dtype)
        stru_out = torch.zeros(B, 1024, device=device, dtype=dtype)

        # Collect present indices
        idx_seq  = [i for i, a in enumerate(aa_list)   if a is not None and len(a) > 0]
        idx_stru = [i for i, s in enumerate(stru_list) if s is not None and len(s) > 0]

        # Batch encode sequences
        if len(idx_seq) > 0:
            seqs = [aa_list[i] for i in idx_seq]
            emb  = self.protein_encoder.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)  # (N,1024)
            emb  = torch.as_tensor(emb, device=device, dtype=dtype)
            seq_out[idx_seq, :] = emb

        # Batch encode structures
        if len(idx_stru) > 0:
            strs = [stru_list[i] for i in idx_stru]
            emb  = self.structure_encoder.get_repr(strs, batch_size=max(1, len(strs)), verbose=False)  # (N,1024)
            emb  = torch.as_tensor(emb, device=device, dtype=dtype)
            stru_out[idx_stru, :] = emb

        return torch.cat([seq_out, stru_out], dim=-1)  # [B, 2048]

    # -------- prefix build ----------
    def build_prefix(self, protein_vec: torch.Tensor) -> torch.Tensor:
        # Always 2048 after encode_protein_batch
        pref = self.proj_2048(protein_vec)  # [B, T, H]
        return pref

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                aa_seq: List[Optional[str]], stru_str: List[Optional[str]]) -> torch.Tensor:
        # Text embeddings (sharding-safe)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # Build protein prefix
        prot_vec = self.encode_protein_batch(aa_seq, stru_str)  # [B, 2048]
        pref = self.build_prefix(prot_vec)                      # [B, P, H]

        # Concat prefix + text
        inputs_embeds  = torch.cat([pref, text_embeds], dim=1)  # [B, P+T, H]
        pref_attn      = torch.ones(inputs_embeds.size(0), pref.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        attn           = torch.cat([pref_attn, attention_mask], dim=1)

        # Extend labels with -100 for prefix
        pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
        new_labels = torch.cat([pad, labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)
        return out

# ------------------------
# Training
# ------------------------

def train(args):
    accelerator = Accelerator()
    set_seed(args.seed)

    # Tokenizer for collator
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_ds = JsonlStream(args.train_file)
    val_ds   = JsonlStream(args.val_file) if args.val_file else None

    collate = PadAndMaskCollator(CollateCfg(tokenizer=tokenizer, max_len=args.max_len))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, collate_fn=collate)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, collate_fn=collate)
                  if val_ds else None)

    # Model
    big = BigProteinQwen(
        model_name=args.model_name,
        protein_config=args.protein_config,
        structure_config=args.structure_config,
        protrek_ckpt=args.protrek_ckpt,
        prot_slot=args.prot_slot,
        stru_slot=args.stru_slot,
        single_token_prefix=args.single_token_prefix,
        prefix_len=args.prefix_len,
        proj_hid=args.proj_hid,
        dropout=args.dropout,
        train_encoders=args.train_encoders,
        dtype_str=args.dtype,
    )

    # Optimizer & schedule
    params = [p for p in big.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Prepare (activates FSDP/ZeRO/mixed precision per accelerate config)
    if val_loader:
        big, optimizer, train_loader, val_loader = accelerator.prepare(big, optimizer, train_loader, val_loader)
    else:
        big, optimizer, train_loader = accelerator.prepare(big, optimizer, train_loader)

    # Steps estimate (placeholder for small runs)
    steps_per_epoch = max(1, 1000 // max(1, args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # Training loop
    big.train(True)
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for it, batch in enumerate(train_loader):
            # move batch tensors to device (Accelerate also handles this)
            for k in ("input_ids", "attention_mask", "labels"):
                batch[k] = batch[k].to(accelerator.device)

            with accelerator.accumulate(big):
                out = big(**batch)
                loss = out.loss / max(1, args.accum_steps)
                if not torch.isfinite(loss):
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            if accelerator.is_main_process and args.log_every and (global_step % args.log_every == 0):
                dt = time.time() - t0
                print(f"[epoch {epoch+1}] step {global_step} | loss={out.loss.item():.4f} | {human_time(dt)}")

            if accelerator.is_main_process and args.save_every and (global_step % args.save_every == 0):
                save_path = os.path.join(args.save_dir, f"ckpt_step{global_step}.pt")
                os.makedirs(args.save_dir, exist_ok=True)
                state = {
                    "llm": accelerator.get_state_dict(big.llm),
                    "proj_1024": accelerator.get_state_dict(big.proj_1024),
                    "proj_2048": accelerator.get_state_dict(big.proj_2048),
                    "protein_encoder": accelerator.get_state_dict(big.protein_encoder) if args.train_encoders else None,
                    "structure_encoder": accelerator.get_state_dict(big.structure_encoder) if args.train_encoders else None,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                torch.save(state, save_path)
                print("Saved:", save_path)
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_path = os.path.join(args.save_dir, "final.pt")
        os.makedirs(args.save_dir, exist_ok=True)
        state = {
            "llm": accelerator.get_state_dict(big.llm),
            "proj_1024": accelerator.get_state_dict(big.proj_1024),
            "proj_2048": accelerator.get_state_dict(big.proj_2048),
            "protein_encoder": accelerator.get_state_dict(big.protein_encoder) if args.train_encoders else None,
            "structure_encoder": accelerator.get_state_dict(big.structure_encoder) if args.train_encoders else None,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
            "epoch": args.epochs,
        }
        torch.save(state, final_path)
        print("Saved FINAL:", final_path)
    accelerator.wait_for_everyone()

# ------------------------
# CLI
# ------------------------

def parse_args():
    p = argparse.ArgumentParser("Protein-conditioned SFT with FSDP/ZeRO")
    # Data
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--val-file", type=str, default=None)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=8)
    # Model
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--protein-config", type=str, required=True)
    p.add_argument("--structure-config", type=str, required=True)
    p.add_argument("--protrek-ckpt", type=str, default=None)
    p.add_argument("--prot-slot", type=int, default=1)
    p.add_argument("--stru-slot", type=int, default=3)
    p.add_argument("--single-token-prefix", action="store_true")
    p.add_argument("--prefix-len", type=int, default=4)
    p.add_argument("--proj-hid", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--train-encoders", action="store_true")
    # Optim
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    # Save/log
    p.add_argument("--save-dir", type=str, default="./runs")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto","fp32","fp16","bf16","float32","float16","bfloat16","default"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
