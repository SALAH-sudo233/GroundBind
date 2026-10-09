This compact mirror preserves the final 15-photo, two-model illustrative attention run. It includes 300 upstream JSON records, 300 selected-token mean attention grids, 300 fresh Omni/JEV records, 105 actual Omni mean grids, frozen ASV policy/decisions and four representative PNGs. The parent adds the 15 original photographs under `images/`. No font files or full raw attention tensors are committed.

Each compact map contains the source JSON SHA256, original full NPZ SHA256, exact selected output-token indices/IDs, merged spatial grid and layer indices. The full raw tensors and all 160/512/1024 budget records remain in the desktop browser package.

To reproduce the four figures from this compact mirror, use the existing environment with NumPy, Pillow and Matplotlib:

```bash
python scripts/build_compact_git_bundle.py --render-representatives --bundle-root /absolute/path/to/this/bundle
```

Outputs appear in `reproduced_representatives/`. Optional `--font-file /absolute/path/to/Times\ New\ Roman.ttf` uses a user-provided local font; otherwise DejaVu Serif is used. Fonts are never installed or committed. The figures use exact compact real attention means, original photographs, fresh candidate boxes and frozen ASV decisions. Blue alpha is 0.67×clip(relative/vmax)^0.7; vmax is the pooled 95th percentile. High values clip, shown by a colorbar arrow. Colors compare relative spatial distributions, not absolute cross-model attention strength.

To rebuild and verify all compact means against the complete desktop raw package:

```bash
python scripts/build_compact_git_bundle.py --root /absolute/path/to/complete/attention_cases
python scripts/build_compact_git_bundle.py --root /absolute/path/to/complete/attention_cases --verify
```

The verifier recomputes every one of the 405 compact maps exactly and checks source hashes and selected indices. The complete desktop root contains `results/`, `verifiers/`, the original `images/`, and the five authoring scripts. This Git mirror intentionally contains their smaller metadata/map counterparts.

Original GPU execution used the existing server environment `/home/u2025141034/.miniconda3/envs/grpo_ayb` (Torch 2.12.1+cu126, Transformers 5.12.1), existing Qwen2.5-VL-7B and Vision-R1 checkpoints, and the original frozen Omni/JEV protocols. Example commands, with scripts located in the complete experiment root:

```bash
RRWR_ATTENTION_ROOT=/home/u2025141034/benchmark/rrwr_20261010_t3_attention/attention
RRWR_PYTHON=/home/u2025141034/.miniconda3/envs/grpo_ayb/bin/python
export LD_LIBRARY_PATH=/home/u2025141034/.miniconda3/envs/grpo_ayb/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_real_attention.py" --root "$RRWR_ATTENTION_ROOT" --model qwen25 --max-new-tokens 160
CUDA_VISIBLE_DEVICES=1 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_real_attention.py" --root "$RRWR_ATTENTION_ROOT" --model visionr1 --max-new-tokens 160
# Only capped records were retried at 512, then only still-capped records at 1024.
CUDA_VISIBLE_DEVICES=0 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_real_attention.py" --root "$RRWR_ATTENTION_ROOT" --model qwen25 --only-truncated --max-new-tokens 512
CUDA_VISIBLE_DEVICES=1 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_real_attention.py" --root "$RRWR_ATTENTION_ROOT" --model visionr1 --only-truncated --max-new-tokens 512
# Repeat these two commands with 1024 for the nine remaining capped records.
"$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/restore_asv_policy.py"
CUDA_VISIBLE_DEVICES=0 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_fresh_verifiers.py" --arm omni
CUDA_VISIBLE_DEVICES=1 "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/run_fresh_verifiers.py" --arm jev
CUDA_VISIBLE_DEVICES="" "$RRWR_PYTHON" "$RRWR_ATTENTION_ROOT/build_attention_visuals.py"
```

ASV policy restoration reads the original source evaluation data and frozen scores, verifies all original 5000 decisions, and never fits on the selected 15 examples. Fresh verifier records are reused only when their upstream JSON hash still matches. These explanatory prompts are qualitative inspection prompts and do not update the paper's standard evaluation metrics.

The attention display uses causal replay rows consuming selected emitted VQA decision / grounding coordinate tokens. Independent Omni attention comes from its final prompt row producing support logits. Before/after ASV use the identical upstream attention array; ASV filters the same candidate and does not rerun the generation model. No Gaussian/bbox-derived or saliency-derived attention is used.
