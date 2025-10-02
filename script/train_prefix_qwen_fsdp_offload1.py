
#!/usr/bin/env python3
"""
Protein-conditioned SFT with a soft prefix on a Causal LM (e.g., Qwen2.5),
now supporting:
- FSDP Full-Shard + CPU offload via 🤗 Accelerate (multi/single GPU)
- Strict FP32 OR bf16 (set in accelerate config)
- Optional 8-bit optimizer (bitsandbytes Adam8bit) or Adafactor
- Train encoders flag (--train-encoders) to backprop through ESM encoders

This script does NOT hardcode mixed precision; Accelerate controls it.
Create a config with `accelerate config` or use the provided YAMLs.

Example (1× A100, bf16, CPU offload, Adam8bit):
  accelerate launch --config_file accelerate_cpu_offload_bf16.yaml \
    train_prefix_qwen_fsdp_offload.py \
      --train-file /data/train.jsonl \
      --model-name Qwen/Qwen2.5-7B-Instruct \
      --protein-config   /weights/ProTrek_35M/esm2_t12_35M_UR50D \
      --structure-config /weights/ProTrek_35M/foldseek_t12_35M \
      --protrek-ckpt     /weights/ProTrek_35M/ProTrek_35M.pt \
      --prot-slot 1 --stru-slot 3 \
      --batch-size 1 --accum-steps 8 \
      --max-len 1024 --prefix-len 4 --prefix-gate 1.0 \
      --epochs 1 --lr 1e-4 --optimizer adam8bit --train-encoders \
      --save-dir /checkpoints/qwen7b_bf16_offload
"""

import os, json, math, argparse, time, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional imports; resolved at runtime
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from transformers import Adafactor
    _HAS_ADAFACTOR = True
except Exception:
    _HAS_ADAFACTOR = False

from accelerate import Accelerator

# Local modules (same folder)
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
# JSONL streaming dataset (sharded by rank)
# ------------------------

class JsonlStream(IterableDataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0) or 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if (i % max(world, 1)) != rank:
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
# Collator: text only (encoders run inside forward for backprop)
# ------------------------

@dataclass
class CollateConfig:
    tokenizer: Any
    max_len: int
    device: torch.device

class PadOnlyTextCollator:
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
            p_keep = min(len(ids_p), len(ids))
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

        aa_list   = [b.get("aa_seq") for b in batch]
        stru_list = [b.get("stru_str") for b in batch]

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
            "aa_seq":         aa_list,
            "stru_str":       stru_list,
        }

# ------------------------
# Encoders (ProTrek slot loader)
# ------------------------

def load_encoders_and_ckpt(protein_config: str,
                           structure_config: str,
                           ckpt_path: Optional[str],
                           prot_slot: int,
                           stru_slot: int):
    ProteinEncoder   = protein_encoder_mod.ProteinEncoder
    StructureEncoder = structure_encoder_mod.StructureEncoder

    prot_enc = ProteinEncoder(protein_config, out_dim=1024, load_pretrained=False)
    stru_enc = StructureEncoder(structure_config, out_dim=1024, load_pretrained=False)

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

    return prot_enc, stru_enc

# ------------------------
# Loss (always computed in float32 for stability)
# ------------------------

def stable_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

# ------------------------
# Training
# ------------------------

def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam8bit":
        if not _HAS_BNB:
            raise RuntimeError("Requested --optimizer adam8bit but bitsandbytes is not installed.")
        return bnb.optim.Adam8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif name == "adafactor":
        if not _HAS_ADAFACTOR:
            raise RuntimeError("Requested --optimizer adafactor but transformers does not provide it.")
        return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def train(args):
    set_seed(args.seed)

    #accelerator = Accelerator()  # accelerates MP + FSDP/offload
    accelerator = Accelerator(split_batches=False)
    device = accelerator.device

    # Determine tensor dtype according to Accelerate mixed precision
    if accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if accelerator.is_main_process:
        print(f"Device: {device} | distributed: {accelerator.num_processes} gpu(s) | mp={accelerator.mixed_precision}")

    # ---- Tokenizer & LLM ----
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
    )
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable()
    hidden_size = llm.config.hidden_size
    if accelerator.is_main_process:
        print(f"Loaded LLM: {args.model_name} | hidden_size={hidden_size}")

    # ---- Encoders ----
    prot_enc, stru_enc = load_encoders_and_ckpt(
        args.protein_config, args.structure_config, args.protrek_ckpt, args.prot_slot, args.stru_slot
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
    )

    # sanity check: projector last layer out_features must be divisible by prefix_len
    _last = projector[-1] if isinstance(projector, nn.Sequential) else None
    if isinstance(_last, nn.Linear):
        assert (_last.out_features % args.prefix_len) == 0, (
            f"projector output ({_last.out_features}) must be divisible by prefix_len ({args.prefix_len})"
        )

    prefix_gate = None
    if args.prefix_gate is not None:
        g = torch.tensor(float(args.prefix_gate))
        prefix_gate = nn.Parameter(g) if args.learnable_gate else g

    # Collect params
    optim_params = list(projector.parameters())
    if prefix_gate is not None and isinstance(prefix_gate, nn.Parameter):
        optim_params += [prefix_gate]
    if not args.freeze_llm:
        optim_params += list(llm.parameters())
    if args.train_encoders:
        optim_params += list(prot_enc.parameters()) + list(stru_enc.parameters())

    optimizer = build_optimizer(args.optimizer, optim_params, args.lr, args.weight_decay)

    # ---- Data ----
    train_ds = JsonlStream(args.train_file)
    val_ds   = JsonlStream(args.val_file) if args.val_file else None

    collate = PadOnlyTextCollator(CollateConfig(
        tokenizer=tok, max_len=args.max_len, device=device
    ))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False, collate_fn=collate)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=False, collate_fn=collate)

    # ---- Prepare with accelerator ----
    modules = [llm, projector]
    if args.train_encoders:
        modules += [prot_enc, stru_enc]

    if args.train_encoders:
        llm, projector, prot_enc, stru_enc, optimizer = accelerator.prepare(
            llm, projector, prot_enc, stru_enc, optimizer
        )
    else:
        llm, projector, optimizer = accelerator.prepare(llm, projector, optimizer)

    def embed_text_ids(model, input_ids):
        # Ensure indices are on the SAME device as the embedding weight (CPU under CPU offload)
        we = model.get_input_embeddings()
        return we(input_ids.to(we.weight.device))

    # ---- Train loop ----
    llm.train(not args.freeze_llm); projector.train()
    if args.train_encoders:
        prot_enc.train(); stru_enc.train()

    global_step = 0
    t0 = time.time()

    def forward_batch(batch) -> Tuple[torch.Tensor, int]:
        input_ids      = batch["input_ids"]
        attn_mask_text = batch["attention_mask"].to(device)
        labels_text    = batch["labels"].to(device)
        B, T = input_ids.shape
        
        text_embeds = embed_text_ids(llm, input_ids)             # now runs on CPU
        text_embeds = text_embeds.to(device, dtype=model_dtype)  # move embeds to GPU/bf16 for the rest of the model

        aa_list   = batch["aa_seq"]
        stru_list = batch["stru_str"]
        need_prot = any(bool(s) for s in aa_list)
        need_stru = any(bool(s) for s in stru_list)

        def zeros_vec(n):
            return torch.zeros(n, 1024, dtype=text_embeds.dtype, device=text_embeds.device)

        if args.train_encoders:
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
        else:
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

        prot_full = zeros_vec(B)
        stru_full = zeros_vec(B)
        ip = 0; is_ = 0
        for i in range(B):
            if aa_list[i]:
                prot_full[i] = prot_vec_list[ip]; ip += 1
            if stru_list[i]:
                stru_full[i] = stru_vec_list[is_]; is_ += 1

        pvec = torch.cat([prot_full, stru_full], dim=1)  # (B, 2048)
        pref = projector(pvec).view(B, args.prefix_len, text_embeds.size(-1))
        if prefix_gate is not None:
            pg = prefix_gate if isinstance(prefix_gate, torch.Tensor) else torch.tensor(prefix_gate, device=pvec.device, dtype=pvec.dtype)
            pref = pref * pg

        inputs_embeds = torch.cat([pref, text_embeds], dim=1)
        attn_mask     = torch.cat([torch.ones(B, args.prefix_len, device=device, dtype=attn_mask_text.dtype),
                                   attn_mask_text], dim=1)
        labels        = torch.cat([torch.full((B, args.prefix_len), -100, device=device, dtype=labels_text.dtype),
                                   labels_text], dim=1)

        out  = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
        loss = stable_ce_loss(out.logits, labels)
        n_valid = int((labels != -100).sum().item())
        return loss, n_valid

    best_val = math.inf
    running = 0.0
    tokens_supervised = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            loss, n_valid = forward_batch(batch)
            if n_valid == 0 or not torch.isfinite(loss):
                if accelerator.is_main_process:
                    print("Skipping batch: no supervised tokens or non-finite loss.")
                continue

            accelerator.backward(loss / args.accum_steps)
            running += float(loss.detach().cpu())
            tokens_supervised += n_valid

            if step % args.accum_steps == 0:
                accelerator.clip_grad_norm_(optim_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if accelerator.is_main_process and (global_step % 50 == 0):
                    avg = running / 50.0
                    print(f"[ep {epoch}] step {global_step} | loss={avg:.4f} | supervised_tokens={tokens_supervised} | time={human_time(time.time()-t0)}")
                    running = 0.0; tokens_supervised = 0

                if accelerator.is_main_process and args.save_every and (global_step % args.save_every == 0):
                    os.makedirs(args.save_dir, exist_ok=True)
                    path = os.path.join(args.save_dir, f"ckpt_step{global_step}.pt")
                    to_save_llm  = accelerator.unwrap_model(llm).state_dict() if not args.freeze_llm else None
                    to_save_proj = accelerator.unwrap_model(projector).state_dict()
                    encoders_blob = None
                    if args.train_encoders:
                        encoders_blob = {
                            "protein":  accelerator.unwrap_model(prot_enc).state_dict() if hasattr(accelerator.unwrap_model(prot_enc), "state_dict") else prot_enc.state_dict(),
                            "structure": accelerator.unwrap_model(stru_enc).state_dict() if hasattr(accelerator.unwrap_model(stru_enc), "state_dict") else stru_enc.state_dict(),
                        }
                    save_state = {
                        "projector": to_save_proj,
                        "llm": to_save_llm,
                        "encoders": encoders_blob,
                        "prefix_gate": (prefix_gate.data if isinstance(prefix_gate, nn.Parameter) else prefix_gate),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    torch.save(save_state, path)
                    print(f"Saved checkpoint to {path}")

                if args.eval_every and (global_step % args.eval_every == 0) and (val_ds is not None):
                    llm.eval(); projector.eval()
                    if args.train_encoders: prot_enc.eval(); stru_enc.eval()
                    with torch.no_grad():
                        vl, vc = 0.0, 0
                        for vb in val_loader:
                            vloss, vn = forward_batch(vb)
                            if vn == 0 or not torch.isfinite(vloss):
                                continue
                            vl += float(vloss.detach().cpu()); vc += 1
                        if accelerator.is_main_process and vc > 0:
                            vavg = vl / vc
                            print(f"  ↳ val loss @ step {global_step}: {vavg:.4f}")
                            best_val = min(best_val, vavg)
                    llm.train(not args.freeze_llm); projector.train()
                    if args.train_encoders: prot_enc.train(); stru_enc.train()

        if accelerator.is_main_process:
            print(f"Finished epoch {epoch}. Elapsed {human_time(time.time()-t0)}")

    # Final save
    if accelerator.is_main_process and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"final.pt")
        to_save_llm  = accelerator.unwrap_model(llm).state_dict() if not args.freeze_llm else None
        to_save_proj = accelerator.unwrap_model(projector).state_dict()
        encoders_blob = None
        if args.train_encoders:
            encoders_blob = {
                "protein":  accelerator.unwrap_model(prot_enc).state_dict() if hasattr(accelerator.unwrap_model(prot_enc), "state_dict") else prot_enc.state_dict(),
                "structure": accelerator.unwrap_model(stru_enc).state_dict() if hasattr(accelerator.unwrap_model(stru_enc), "state_dict") else stru_enc.state_dict(),
            }
        save_state = {
            "projector": to_save_proj,
            "llm": to_save_llm,
            "encoders": encoders_blob,
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
    p = argparse.ArgumentParser(description="Protein-conditioned SFT with soft prefix (FSDP Offload + optional 8-bit)")
    # Data
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--val-file",   type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--max-len", type=int, default=1024)
    # Model
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, choices=["adamw", "adam8bit", "adafactor"], default="adamw",
                   help="Choose AdamW (default), bitsandbytes Adam8bit, or Adafactor")
    # Save/eval
    p.add_argument("--save-dir", type=str, default="./runs_offload")
    p.add_argument("--save-every", type=int, default=0, help="save every N optimizer steps (0=disabled)")
    p.add_argument("--eval-every", type=int, default=0, help="eval every N optimizer steps (0=disabled)")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
