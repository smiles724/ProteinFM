#!/usr/bin/env python3
"""
ProTrek Encoders-Only Pipeline
------------------------------
Read FASTA sequences and/or PDB/CIF structures and write 1024-D embeddings
using the standalone encoders + the tri-modal checkpoint (to load trained heads).

Example usage (WSL/Linux):
  python protrek_embed_pipeline.py \
    --checkpoint weights/ProTrek_650M/ProTrek_650M.pt \
    --protein-config weights/ProTrek_650M/esm2_t33_650M_UR50D \
    --structure-config weights/ProTrek_650M/foldseek_t30_150M \
    --foldseek-bin bin/foldseek \
    --fasta-dir data/fasta \
    --struct-dir data/structures \
    --chains A \
    --out-dir outputs/protrek_embeddings \
    --batch-size 16 \
    --device auto

Outputs:
  - *.npy files for embeddings
  - manifest.csv with metadata and (if available) seq↔struct similarity

Notes:
  - Structure encoder expects Foldseek "3Di" strings; we generate them via foldseek_util.get_struc_seq(...).
  - For PDB/CIF inputs, we can also produce sequence embeddings from the AA returned by Foldseek.
  - FASTA parsing is lightweight and does not require Biopython.
"""

from __future__ import annotations

import os, sys, csv, argparse, math, json, time
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import torch


from model.protein_encoder import ProteinEncoder
from model.structure_encoder import StructureEncoder
from model.foldseek_util import get_struc_seq  # expects (foldseek_bin, pdb_or_cif_path, chains) -> {chain: (aa_seq, struc_seq, combined)}



def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_fasta(path: Path) -> bool:
    return path.suffix.lower() in {".fa", ".fasta", ".faa"}


def is_struct(path: Path) -> bool:
    return path.suffix.lower() in {".pdb", ".cif"}


def iter_files(root: Path, exts: Optional[set] = None) -> Iterable[Path]:
    if root.is_file():
        if exts is None or root.suffix.lower() in exts:
            yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and (exts is None or p.suffix.lower() in exts):
            yield p


def chunked(lst: List, n: int) -> Iterable[List]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def read_fasta(path: Path) -> Dict[str, str]:
    """Minimal FASTA reader: returns {header: sequence} (headers without '>')."""
    d: Dict[str, str] = {}
    header = None
    seq_parts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    d[header] = "".join(seq_parts).replace(" ", "").replace("\t", "")
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            d[header] = "".join(seq_parts).replace(" ", "").replace("\t", "")
    return d


def save_npy(arr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    return str(path)


def ensure_on_path(repo_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))




def load_submodule_from_trimo_ckpt(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str):
    """Load keys that start with a given prefix into a submodule (strict=False)."""
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sub, strict=False)
    log(f"Loaded submodule '{prefix}' -> params: {len(sub)} | missing: {len(missing)} | unexpected: {len(unexpected)}")
    if len(sub) == 0:
        log(f"WARNING: no keys with prefix '{prefix}'. Trying full load with strict=False.")
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        log(f"Fallback load -> missing: {len(missing)} | unexpected: {len(unexpected)}")


def load_encoders(
    protein_config: str,
    structure_config: str,
    checkpoint_path: str,
    device: str = "auto",
    out_dim: int = 1024,
) -> Tuple[ProteinEncoder, StructureEncoder, torch.device]:
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    log(f"Using device: {device_t} (CUDA avail: {torch.cuda.is_available()})")

    # Build encoders
    prot_enc = ProteinEncoder(protein_config, out_dim=out_dim, load_pretrained=True).to(device_t).eval()
    stru_enc = StructureEncoder(structure_config, out_dim=out_dim).to(device_t).eval()

    # Load tri-modal checkpoint to populate the trained heads/backbones
    log(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    load_submodule_from_trimo_ckpt(prot_enc, state_dict, prefix="protein_encoder.")
    load_submodule_from_trimo_ckpt(stru_enc, state_dict, prefix="structure_encoder.")

    return prot_enc, stru_enc, device_t



@torch.no_grad()
def embed_sequences(prot_enc: ProteinEncoder, seqs: List[str], device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Return array [N, 1024] of L2-normalized embeddings."""
    outs = []
    for chunk in chunked(seqs, batch_size):
        emb = prot_enc.get_repr(chunk)  # expected [B, 1024], already normalized
        emb = emb.to(device)
        outs.append(emb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1024), dtype=np.float32)


@torch.no_grad()
def embed_structures(stru_enc: StructureEncoder, foldseek_strs: List[str], device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Return array [N, 1024] of L2-normalized embeddings for Foldseek 3Di strings (lowercased)."""
    outs = []
    for chunk in chunked(foldseek_strs, batch_size):
        # ensure lower-case per README example
        chunk = [s.lower() for s in chunk]
        emb = stru_enc.get_repr(chunk)  # expected [B, 1024], already normalized
        emb = emb.to(device)
        outs.append(emb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1024), dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:

    return float(np.dot(a.ravel(), b.ravel()))




def main():
    ap = argparse.ArgumentParser(description="ProTrek encoders-only pipeline for FASTA + PDB/CIF to embeddings.")
    ap.add_argument("--checkpoint", required=True, help="Path to ProTrek tri-modal checkpoint (e.g., weights/.../ProTrek_650M.pt)")
    ap.add_argument("--protein-config", required=True, help="Path to HF config dir for protein encoder (e.g., weights/.../esm2_t33_650M_UR50D)")
    ap.add_argument("--structure-config", required=True, help="Path to HF config dir for structure encoder (e.g., weights/.../foldseek_t30_150M)")

    ap.add_argument("--fasta-dir",  type=str, default=None, help="Folder or file with FASTA (*.fa, *.fasta, *.faa)")
    ap.add_argument("--fasta-file", type=str, default=None, help="Specific FASTA file (overrides --fasta-dir if both given)")

    ap.add_argument("--struct-dir",  type=str, default=None, help="Folder or file with structures (*.pdb, *.cif)")
    ap.add_argument("--struct-file", type=str, default=None, help="Specific PDB/CIF file (overrides --struct-dir if both given)")
    ap.add_argument("--foldseek-bin", type=str, default=None, help="Path to foldseek binary for structure conversion (required if struct inputs given)")
    ap.add_argument("--chains", type=str, default="A", help='Comma-separated chain IDs to process from structures (default "A")')
    ap.add_argument("--threads", type=int, default=1, help="Threads for Foldseek")

    ap.add_argument("--out-dir", type=str, default="outputs/protrek_embeddings", help="Output directory")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for encoder forward")
    ap.add_argument("--device", type=str, default="auto", help='"auto", "cuda", or "cpu"')
    ap.add_argument("--write-manifest", action="store_true", help="Write manifest.csv with metadata and similarities")
    args = ap.parse_args()

    repo_dir = Path.cwd()
    ensure_on_path(repo_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prot_enc, stru_enc, device_t = load_encoders(
        protein_config=args.protein_config,
        structure_config=args.structure_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        out_dim=1024,
    )

    manifest_rows = []

    # ---------- FASTA processing ----------
    fasta_targets: List[Path] = []
    if args.fasta_file:
        fasta_targets = [Path(args.fasta_file)]
    elif args.fasta_dir:
        fasta_targets = list(iter_files(Path(args.fasta_dir), exts={".fa", ".fasta", ".faa"}))

    if fasta_targets:
        log(f"Found {len(fasta_targets)} FASTA target(s).")
        for fpath in fasta_targets:
            log(f"Reading FASTA: {fpath}")
            entries = read_fasta(fpath)
            headers = list(entries.keys())
            seqs = [entries[h] for h in headers]
            if len(seqs) == 0:
                log(f"[warn] No sequences in {fpath}")
                continue
            log(f"Embedding {len(seqs)} sequences from {fpath} ...")
            embs = embed_sequences(prot_enc, seqs, device_t, batch_size=args.batch_size)
            for header, seq, emb in zip(headers, seqs, embs):
                stem = f"{fpath.stem}__{header.replace(' ', '_')[:60]}"
                npy_path = out_dir / f"{stem}_seq_emb.npy"
                save_npy(emb.astype(np.float32), npy_path)
                manifest_rows.append({
                    "type": "fasta",
                    "file": str(fpath),
                    "entry": header,
                    "chain": "",
                    "aa_len": len(seq),
                    "foldseek_len": "",
                    "seq_emb_path": str(npy_path),
                    "stru_emb_path": "",
                    "seq_stru_similarity": ""
                })

    # ---------- Structure processing ----------
    struct_targets: List[Path] = []
    if args.struct_file:
        struct_targets = [Path(args.struct_file)]
    elif args.struct_dir:
        struct_targets = list(iter_files(Path(args.struct_dir), exts={".pdb", ".cif"}))

    if struct_targets:
        if not args.foldseek_bin:
            raise SystemExit("ERROR: --foldseek-bin is required for structure inputs.")
        chains = [c for c in args.chains.split(",") if c]
        log(f"Found {len(struct_targets)} structure target(s). Chains: {chains}")

        for fpath in struct_targets:
            log(f"Foldseek → 3Di: {fpath}")
            try:
                seqs_dict = get_struc_seq(args.foldseek_bin, str(fpath), chains)
            except Exception as e:
                log(f"[skip] {fpath} -> {e}")
                continue

            for ch, tpl in seqs_dict.items():
                if not tpl or len(tpl) < 2:
                    continue
                aa_seq, fseek_seq = tpl[0], tpl[1].lower()

                # Embed (sequence and structure)
                seq_emb  = embed_sequences(prot_enc, [aa_seq],  device_t, batch_size=1)
                stru_emb = embed_structures(stru_enc, [fseek_seq], device_t, batch_size=1)

                # Save
                stem = f"{fpath.stem}_chain{ch}"
                seq_path  = out_dir / f"{stem}_seq_emb.npy"
                stru_path = out_dir / f"{stem}_stru_emb.npy"
                save_npy(seq_emb[0].astype(np.float32),  seq_path)
                save_npy(stru_emb[0].astype(np.float32), stru_path)

                # Similarity (cosine == dot because encoders normalize)
                sim = cosine_sim(seq_emb[0], stru_emb[0])

                manifest_rows.append({
                    "type": "structure",
                    "file": str(fpath),
                    "entry": "",
                    "chain": ch,
                    "aa_len": len(aa_seq),
                    "foldseek_len": len(fseek_seq),
                    "seq_emb_path": str(seq_path),
                    "stru_emb_path": str(stru_path),
                    "seq_stru_similarity": f"{sim:.6f}"
                })

    # ---------- Manifest ----------
    if args.write_manifest and manifest_rows:
        csv_path = out_dir / "manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "type","file","entry","chain","aa_len","foldseek_len",
                "seq_emb_path","stru_emb_path","seq_stru_similarity"
            ])
            writer.writeheader()
            writer.writerows(manifest_rows)
        log(f"Wrote manifest: {csv_path}")

    log("Done.")


if __name__ == "__main__":
    main()
