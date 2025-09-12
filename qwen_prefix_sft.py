#!/usr/bin/env python3
# (Full script content repeated due to state reset)
from __future__ import annotations
import os, json, math, random, argparse
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    AdamW,
)

class JsonlSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                obj = json.loads(line)
                self.rows.append(obj)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt   = ex.get("prompt", "")
        response = ex.get("response", "")
        seq_emb_path  = ex.get("seq_emb", None)
        stru_emb_path = ex.get("stru_emb", None)

        text_prompt   = prompt.strip()
        text_response = response.strip()

        prompt_ids   = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        resp_ids     = self.tokenizer(text_response, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        seq_vec  = np.load(seq_emb_path).astype(np.float32)  if seq_emb_path  else None
        stru_vec = np.load(stru_emb_path).astype(np.float32) if stru_emb_path else None

        return {
            "prompt_ids": prompt_ids,
            "resp_ids": resp_ids,
            "seq_vec": seq_vec,
            "stru_vec": stru_vec,
        }

@dataclass
class CollatorCfg:
    tokenizer: any
    model: any
    prefix_len: int = 4
    proj_hid: int = 1024
    dropout: float = 0.1
    max_len: int = 3072

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, prefix_len: int = 4, proj_hid: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_len  = prefix_len
        out_dim = hidden_size * prefix_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y.view(x.size(0), self.prefix_len, -1)
        return y

class PrefixCollator:
    def __init__(self, cfg: CollatorCfg):
        self.tok = cfg.tokenizer
        self.model = cfg.model
        self.prefix_len = cfg.prefix_len
        self.max_len = cfg.max_len

        hidden_size = self.model.config.hidden_size
        self.proj_1024 = PrefixProjector(1024, hidden_size, cfg.prefix_len, cfg.proj_hid, cfg.dropout).to(self.model.device)
        self.proj_2048 = PrefixProjector(2048, hidden_size, cfg.prefix_len, cfg.proj_hid, cfg.dropout).to(self.model.device)

    def to(self, device):
        self.proj_1024.to(device)
        self.proj_2048.to(device)

    def _project_prefix(self, seq_vec, stru_vec) -> torch.Tensor:
        vecs = []
        for s, t in zip(seq_vec, stru_vec):
            if s is None and t is None:
                vecs.append(np.zeros((1024,), dtype=np.float32))
            elif s is None:
                vecs.append(t)
            elif t is None:
                vecs.append(s)
            else:
                vecs.append(np.concatenate([s, t], axis=-1))
        arr = np.stack(vecs, axis=0)
        ten = torch.from_numpy(arr).to(self.model.device)

        if arr.shape[1] == 1024:
            return self.proj_1024(ten)
        elif arr.shape[1] == 2048:
            return self.proj_2048(ten)
        else:
            raise ValueError(f"Unexpected vector dim {arr.shape[1]} (expected 1024 or 2048).")

    def __call__(self, batch: List[Dict]):
        prompt_ids_list = [b["prompt_ids"] for b in batch]
        resp_ids_list   = [b["resp_ids"] for b in batch]

        seq_vec = [b["seq_vec"] for b in batch]
        stru_vec = [b["stru_vec"] for b in batch]
        prefix_embeds = self._project_prefix(seq_vec, stru_vec)

        input_ids_list = []
        for p_ids, r_ids in zip(prompt_ids_list, resp_ids_list):
            max_txt = self.max_len - self.prefix_len - 1
            ids = p_ids + [self.tok.eos_token_id] + r_ids
            ids = ids[:max_txt]
            input_ids_list.append(ids)

        max_len_ids = max(len(x) for x in input_ids_list) if input_ids_list else 1
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids = []
        attn_mask = []
        for ids in input_ids_list:
            pad_n = max_len_ids - len(ids)
            input_ids.append(ids + [pad_id]*pad_n)
            attn_mask.append([1]*len(ids) + [0]*pad_n)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.model.device)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=self.model.device)

        tok_embeds = self.model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)

        prefix_mask = torch.ones((inputs_embeds.size(0), self.prefix_len), dtype=attn_mask.dtype, device=attn_mask.device)
        full_attn = torch.cat([prefix_mask, attn_mask], dim=1)

        labels = input_ids.clone()
        eos_id = self.tok.eos_token_id
        for i in range(labels.size(0)):
            row = labels[i].tolist()
            try:
                eos_idx = row.index(eos_id)
            except ValueError:
                eos_idx = len(row) - 1
            for j in range(0, eos_idx + 1):
                labels[i, j] = -100

        prefix_labels = torch.full((labels.size(0), self.prefix_len), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attn,
            "labels": full_labels,
        }

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.device != "auto":
        model.to(args.device)

    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type=TaskType.CAUSAL_LM,
                target_modules=args.lora_targets.split(",") if args.lora_targets else None,
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()
        except Exception as e:
            print("PEFT not installed or failed to init LoRA; proceeding without LoRA.\n", e)

    train_data = JsonlSFTDataset(args.train_file, tokenizer, max_len=args.max_len)
    val_data   = JsonlSFTDataset(args.val_file, tokenizer, max_len=args.max_len) if args.val_file else None

    collate = PrefixCollator(CollatorCfg(
        tokenizer=tokenizer, model=model,
        prefix_len=args.prefix_len, proj_hid=args.proj_hid, dropout=args.dropout,
        max_len=args.max_len
    ))
    collate.to(model.device)

    train_loader = DataLoader(train_data, batch_size=args.bsz, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_data, batch_size=args.bsz, shuffle=False, collate_fn=collate) if val_data else None

    params = list(model.parameters()) + list(collate.proj_1024.parameters()) + list(collate.proj_2048.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.accum))
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(model.device.type == "cuda" and not args.force_fp32))

    global_step = 0
    best_val = None

    for epoch in range(args.epochs):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=(model.device.type == "cuda" and not args.force_fp32)):
                out = model(**batch)
                loss = out.loss / args.accum

            scaler.scale(loss).backward()
            running += loss.item()

            if (step + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % max(1, args.log_every) == 0:
                    print(f"[epoch {epoch+1}] step {global_step}/{total_steps}  loss={running:.4f}")
                    running = 0.0

        if val_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(**batch)
                    losses.append(out.loss.item())
            val_loss = float(np.mean(losses)) if losses else None
            print(f"[epoch {epoch+1}] val_loss={val_loss}")
            model.train()

            if best_val is None or (val_loss is not None and val_loss < best_val):
                best_val = val_loss
                model.save_pretrained(os.path.join(args.output_dir, "model_best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "model_best"))
                torch.save({
                    "proj_1024": collate.proj_1024.state_dict(),
                    "proj_2048": collate.proj_2048.state_dict(),
                }, os.path.join(args.output_dir, "model_best", "protein_prefix_projectors.pt"))

        model.save_pretrained(os.path.join(args.output_dir, f"model_epoch{epoch+1}"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, f"model_epoch{epoch+1}"))
        torch.save({
            "proj_1024": collate.proj_1024.state_dict(),
            "proj_2048": collate.proj_2048.state_dict(),
        }, os.path.join(args.output_dir, f"model_epoch{epoch+1}", "protein_prefix_projectors.pt"))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-file", type=str, required=True)
    ap.add_argument("--val-file", type=str, default=None)

    ap.add_argument("--output-dir", type=str, default="runs/qwen25_prefix_sft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--force-fp32", action="store_true")

    ap.add_argument("--prefix-len", type=int, default=4)
    ap.add_argument("--proj-hid", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--log-every", type=int, default=20)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
