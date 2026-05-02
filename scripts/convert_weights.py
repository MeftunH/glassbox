#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights to the glassbox `.glx` flat format.

The output layout is:

    [u64 little-endian header_len]
    [utf8 JSON header of length header_len]
    [raw payload]

Where the header is:

    {
        "magic": "GLX1",
        "version": 1,
        "architecture": "gpt2",
        "config": { ...ModelConfig... },
        "tokenizer_blob": "<json string>",
        "tensors": [
            {"name": "...", "dtype": "f32", "shape": [...], "offset": ..., "byte_len": ...},
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

GLX_MAGIC = b"GLX1"
GLX_VERSION = 1


def required(*pkgs):
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            missing.append(p)
    if missing:
        sys.stderr.write(
            f"missing python packages: {missing}\n"
            f"install with: pip install transformers tokenizers safetensors\n"
        )
        sys.exit(2)


def load_gpt2(name: str):
    required("transformers")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(name)
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    return model, tokenizer


def gpt2_tensors(model) -> dict[str, np.ndarray]:
    sd = model.state_dict()
    out: dict[str, np.ndarray] = {}

    out["wte"] = sd["transformer.wte.weight"].detach().cpu().numpy().astype(np.float32)
    out["wpe"] = sd["transformer.wpe.weight"].detach().cpu().numpy().astype(np.float32)

    n_layer = model.config.n_layer
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        out[f"h.{i}.ln_1.g"] = sd[f"{prefix}.ln_1.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.ln_1.b"] = sd[f"{prefix}.ln_1.bias"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.attn.c_attn.w"] = sd[f"{prefix}.attn.c_attn.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.attn.c_attn.b"] = sd[f"{prefix}.attn.c_attn.bias"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.attn.c_proj.w"] = sd[f"{prefix}.attn.c_proj.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.attn.c_proj.b"] = sd[f"{prefix}.attn.c_proj.bias"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.ln_2.g"] = sd[f"{prefix}.ln_2.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.ln_2.b"] = sd[f"{prefix}.ln_2.bias"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.mlp.c_fc.w"] = sd[f"{prefix}.mlp.c_fc.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.mlp.c_fc.b"] = sd[f"{prefix}.mlp.c_fc.bias"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.mlp.c_proj.w"] = sd[f"{prefix}.mlp.c_proj.weight"].detach().cpu().numpy().astype(np.float32)
        out[f"h.{i}.mlp.c_proj.b"] = sd[f"{prefix}.mlp.c_proj.bias"].detach().cpu().numpy().astype(np.float32)

    out["ln_f.g"] = sd["transformer.ln_f.weight"].detach().cpu().numpy().astype(np.float32)
    out["ln_f.b"] = sd["transformer.ln_f.bias"].detach().cpu().numpy().astype(np.float32)

    return out


def tokenizer_blob(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    merges_path = Path(tokenizer.vocab_files_names.get("merges_file", "merges.txt"))
    merges: list[tuple[str, str]] = []
    if hasattr(tokenizer, "backend_tokenizer"):
        try:
            data = json.loads(tokenizer.backend_tokenizer.to_str())
            raw = data.get("model", {}).get("merges", [])
            for line in raw:
                if isinstance(line, str):
                    parts = line.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                elif isinstance(line, list) and len(line) == 2:
                    merges.append((line[0], line[1]))
        except Exception:
            pass

    return json.dumps({"vocab": vocab, "merges": merges}, ensure_ascii=False)


def model_config(model) -> dict[str, Any]:
    c = model.config
    return {
        "architecture": "gpt2",
        "vocab_size": c.vocab_size,
        "n_positions": c.n_positions,
        "n_embd": c.n_embd,
        "n_layer": c.n_layer,
        "n_head": c.n_head,
        "layer_norm_epsilon": float(c.layer_norm_epsilon),
    }


def write_glx(out_path: Path, config: dict, tokenizer_str: str, tensors: dict[str, np.ndarray]) -> None:
    payload = io.BytesIO()
    entries = []
    for name, arr in tensors.items():
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        offset = payload.tell()
        payload.write(arr.tobytes())
        entries.append({
            "name": name,
            "dtype": "f32",
            "shape": list(arr.shape),
            "offset": offset,
            "byte_len": arr.nbytes,
        })

    header = {
        "magic": "GLX1",
        "version": GLX_VERSION,
        "architecture": "gpt2",
        "config": config,
        "tokenizer_blob": tokenizer_str,
        "tensors": entries,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(GLX_MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model name")
    parser.add_argument("--out", required=True, type=Path, help="output .glx path")
    args = parser.parse_args()

    model, tokenizer = load_gpt2(args.model)
    cfg = model_config(model)
    tensors = gpt2_tensors(model)
    tok_blob = tokenizer_blob(tokenizer)

    write_glx(args.out, cfg, tok_blob, tensors)

    total_params = sum(t.size for t in tensors.values())
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(
        f"wrote {args.out}: {len(tensors)} tensors, "
        f"{total_params / 1e6:.1f}M params, {total_bytes / 1e6:.1f} MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
