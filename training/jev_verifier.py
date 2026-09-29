#!/usr/bin/env python3
"""A JEV-style calibrated verifier: image -> P(keep), no binding, no generation.

Architecture copied from Open-Jev-2B (Apache-2.0, ZefanCai/Open-Jev-2B), whose
pinned base revision 15852e8c16360a2fea060d615a32b45270f8a8fc is the SAME snapshot
our GRPO verifier uses:
    base (frozen, multimodal) + LoRA + a (1, hidden) scalar head + one temperature
    objective: NLL + brier_weight * Brier      (brier_weight = 0.1)
Their released adapter is NOT loaded: it only touches text-tower modules and its head
never saw images, so it cannot judge a box. We reuse the RECIPE, not the weights.

Why this instead of more GRPO: GRPO produced a confident binary decider whose
confidence saturates at {0,1}, leaving no operating point — the recorded reason the
FNR<=3% gate looked unsatisfiable. Brier is a strictly proper scoring rule, so
overconfidence is penalised during training rather than patched afterwards.

The head reads the LAST prompt position (no tokens are generated), so one forward
pass per item. Target is the decision label only:
    y = 1 if the phrase is true of the region (KEEP) else 0

Splits: train on the 2000-image training pool (16000 rows, same file the GRPO arms
used). Calibration temperature is fitted on a held-out slice of that pool. The
500-dev set is touched only at eval time.
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/"
    "15852e8c16360a2fea060d615a32b45270f8a8fc")
W = os.path.expanduser("~/SVD/grpo_verifier/")
PROMPT = ('Inspect the image region [{x0},{y0},{x1},{y1}]. '
          'Determine whether the phrase "{q}" is fully true of that region.')


def build_inputs(proc, img_path, text, dev):
    msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    prompt_text = proc.apply_chat_template([msg], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
    pil = Image.open(img_path).convert("RGB")
    inp = proc(text=[prompt_text], images=[pil], return_tensors="pt")
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}


class JevVerifier(nn.Module):
    """Frozen-ish multimodal base + LoRA + scalar decision head on the last position."""

    def __init__(self, base_path, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.base = AutoModelForImageTextToText.from_pretrained(
            base_path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
        # The base has 24 layers: 18 linear_attention (in_proj_qkv / out_proj) and
        # only 6 full_attention (q/k/v/o_proj). Listing just q/k/v/o attaches LoRA to
        # 6 of 24 layers and silently freezes 75% of the stack — the first run did
        # exactly that (48 adapter tensors instead of 120) and collapsed to a constant
        # REJECT. Open-Jev's own config lists all six names; match it.
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                         bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "in_proj_qkv", "out_proj"])
        self.base = get_peft_model(self.base, cfg)
        hidden = self.base.config.get_text_config().hidden_size
        # fp32 head: a bf16 linear on top of a near-zero init produces logits whose
        # quantisation is coarse relative to their magnitude, which showed up in the
        # smoke run as a degenerate temperature fit.
        self.head = nn.Linear(hidden, 1, dtype=torch.float32).to("cuda:0")
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        self.log_t = nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cuda:0"))

    def logit(self, inputs):
        out = self.base(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1, :]          # last prompt position
        return self.head(h.float()).squeeze(-1)

    def prob(self, inputs, calibrated=True):
        z = self.logit(inputs)
        if calibrated:
            z = z / self.log_t.exp()
        return torch.sigmoid(z)


def loss_fn(z, y, brier_w=0.1):
    """NLL + brier_w * Brier, Open-Jev's objective."""
    p = torch.sigmoid(z)
    nll = nn.functional.binary_cross_entropy_with_logits(z, y)
    brier = ((p - y) ** 2).mean()
    return nll + brier_w * brier, nll.item(), brier.item()


def load_train(path, limit=0):
    rows = []
    for line in open(path, encoding="utf8"):
        if not line.strip():
            continue
        d = json.loads(line)
        content = d["messages"][0]["content"]
        # the region and phrase are already inside the prompt text; reuse verbatim
        rows.append(dict(text=content.replace("<image>", "").strip(),
                         image=d["images"][0],
                         y=1.0 if d["solution"] == "KEEP" else 0.0,
                         htype=d["htype"], sid=d["sid"]))
    random.Random(63).shuffle(rows)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=W + "q35_2b_train2000_v3_audit_phrase.jsonl")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--brier_w", type=float, default=0.1)
    ap.add_argument("--n_cal", type=int, default=512)
    ap.add_argument("--out", default=W + "runs/jev_verifier")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(63)
    random.seed(63)

    rows = load_train(args.train, args.limit)
    cal = rows[:args.n_cal]
    tr = rows[args.n_cal:]
    print("train {}  calibration {}".format(len(tr), len(cal)), flush=True)
    print("train label balance: {:.3f}".format(sum(r["y"] for r in tr) / len(tr)), flush=True)

    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True,
                                        min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    model = JevVerifier(BASE)
    model.base.print_trainable_parameters()

    opt = torch.optim.AdamW([
        {"params": [p for p in model.base.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.0)

    model.train()
    step = 0
    i = 0
    run_nll = run_br = 0.0
    seen = 0
    while step < args.steps:
        opt.zero_grad()
        for _ in range(args.accum):
            zs, ys = [], []
            for _ in range(args.bs):
                r = tr[i % len(tr)]
                i += 1
                try:
                    inp = build_inputs(proc, r["image"], r["text"], "cuda:0")
                    zs.append(model.logit(inp))
                    ys.append(r["y"])
                except Exception as e:
                    print("  skip: {}".format(repr(e)[:80]), flush=True)
            if not zs:
                continue
            z = torch.cat(zs)
            y = torch.tensor(ys, dtype=torch.float32, device=z.device)
            loss, nll, br = loss_fn(z, y, args.brier_w)
            (loss / args.accum).backward()
            run_nll += nll
            run_br += br
            seen += 1
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.base.parameters() if p.requires_grad] +
            list(model.head.parameters()), 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            print("step {}/{}  NLL {:.4f}  Brier {:.4f}".format(
                step, args.steps, run_nll / max(1, seen), run_br / max(1, seen)), flush=True)
            run_nll = run_br = 0.0
            seen = 0
        if step % 250 == 0 or step == args.steps:
            torch.save({"head": model.head.state_dict(),
                        "log_t": model.log_t.detach().cpu()},
                       os.path.join(args.out, "head_step{}.pt".format(step)))
            model.base.save_pretrained(os.path.join(args.out, "adapter_step{}".format(step)))

    # ---- temperature on the held-out calibration slice ----
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for r in cal:
            try:
                inp = build_inputs(proc, r["image"], r["text"], "cuda:0")
                zs.append(model.logit(inp).item())
                ys.append(r["y"])
            except Exception:
                pass
    print("calibration items scored: {}".format(len(zs)), flush=True)

    def nll_at(t):
        s = 0.0
        for z, y in zip(zs, ys):
            p = min(1 - 1e-9, max(1e-9, 1.0 / (1.0 + math.exp(-z / t))))
            s -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return s / len(zs)

    # Grid must reach well below 1.0: a near-zero-init head gives tiny logits, so the
    # NLL optimum can sit at a very small temperature (sharpening). The smoke run hit
    # the old 0.25 floor, which means the floor was the answer, not the optimum.
    best_t, best = 1.0, None
    t = 0.02
    while t <= 8.0:
        v = nll_at(t)
        if best is None or v < best:
            best, best_t = v, t
        t *= 1.06

    def ece(t, bins=10):
        ps = [1.0 / (1.0 + math.exp(-z / t)) for z in zs]
        tot = 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = [(p, y) for p, y in zip(ps, ys)
                   if p >= lo and (p < hi or (b == bins - 1 and p <= hi))]
            if sel:
                tot += len(sel) / len(ps) * abs(
                    sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
        return tot

    print("temperature {:.2f}  NLL {:.4f} -> {:.4f}   ECE {:.4f} -> {:.4f}".format(
        best_t, nll_at(1.0), best, ece(1.0), ece(best_t)), flush=True)

    with open(os.path.join(args.out, "temperature.json"), "w", encoding="utf8") as f:
        json.dump({"temperature": best_t, "split": "calibration", "n": len(zs),
                   "nll_before": nll_at(1.0), "nll_after": best,
                   "ece_before": ece(1.0), "ece_after": ece(best_t)}, f, indent=1)
    torch.save({"head": model.head.state_dict(),
                "temperature": best_t}, os.path.join(args.out, "head_final.pt"))
    model.base.save_pretrained(os.path.join(args.out, "adapter_final"))
    print("saved to " + args.out, flush=True)


if __name__ == "__main__":
    main()
