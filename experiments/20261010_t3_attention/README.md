# Semantic T3 scoring and attention cases — 2026-10-10

This supplement adds EOH/ROH false-assertion rates for query-free captions from all 13 benchmark models and records new attention case runs on `vlm1`. Existing benchmark rates and original caption outputs are preserved.

## Caption results

The archived subset contains 500 original descriptions per model, 6,500 descriptions total. Four annotated negative semantic units per image produce 26,000 textual-assertion judgments: 1,000 EOH and 1,000 ROH units per model. The generator receives only the image. A separate fixed Qwen3.5-9B evaluator checks whether the resulting caption asserts the edited semantic proposition, with target identity, properties, and relation roles bound explicitly.

`caption/t3_main_table_rows.csv` contains all integer counts and the 13 model rows. Equal-model EOH/ROH FAR means are 7.725/2.900% for General and 8.733333/3.200% for RL. FAR here means false assertion; expression-verification FAR means false acceptance. The source archives retain the original caption and object-level metrics.

The locked scorer attained 63/64 agreement on the independent, predefined held-out semantic controls. Development controls and the earlier failed prompt are archived separately. Complete inputs, outputs, source annotations and caption text are published as gzip JSONL files; no duplicated raw shards are required to inspect the results. `release_manifest.json` lists the files shipped in this supplement. The original full-run SHA audit also lists server archival files omitted from the Git package.

## Reproduce the published counts

From this directory, Python's standard library is sufficient:

```sh
python caption/reproduce_archive.py verify
```

To unpack the full archive and restore the original four shard files without a GPU:

```sh
python caption/reproduce_archive.py materialize --out replay --restore-shards
```

To rerun the locked judge, materialize into a fresh directory without `--restore-shards`, then use the pinned Qwen3.5-9B snapshot (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`) with PyTorch 2.12.1 and Transformers 5.12.1:

```sh
python caption/reproduce_archive.py materialize --out new_replay
CUDA_VISIBLE_DEVICES=0 python caption/reproduce_archive.py rescore --out new_replay --model-path /path/to/the/pinned/snapshot --shard 0 --nshards 4 --batch 8
```

Repeat `--shard 1`, `2`, and `3` on available GPUs. The wrapper substitutes only the local checkpoint location and calls the original locked scorer; it leaves the scorer file and semantic protocol unchanged. `caption/launch_all.py` preserves the actual server launch used for this run.

## Attention cases

The case run selects 15 source images, evaluates Qwen2.5-VL and Vision-R1, and preserves VQA and grounding outputs for one supported and four edited queries per image. Attention comes from the last four decoder layers during causal replay of the actual greedy-generated token sequence; selected answer or coordinate tokens retain all heads. Image token order and merged patch grids are checked against the installed model source.

ASV reuses each new upstream candidate. Its heads and thresholds are reconstructed from the original calibration records and checked against all 5,000 original decisions for the two backbones. The new candidates are rescored by the two frozen verifiers, including the original predicate alternatives. Before/after upstream attention therefore uses the same array; independent OmniVerifier attention is labeled as a separate column. The actual run includes documented generation budgets and preserved capped outputs.

The Desktop gallery contains all cases, overlays, raw attention tensors and observations for author review. This repository supplement contains the compact records and derived patch maps, reproduction scripts, and representative images. The case observations are not inserted into the manuscript.

To recreate and verify the 405 compact maps from the full gallery archive, use a root containing its original photos, JSON records and raw NPZ tensors:

```sh
python attention/scripts/build_compact_git_bundle.py --root /path/to/full/gallery
python attention/scripts/build_compact_git_bundle.py --root /path/to/full/gallery --verify
```

This checks the selected emitted-token indices, mean patch weights and both source-record and raw-tensor SHA256 values. The release includes its 15 source photographs separately in `images/`; full raw tensors remain in the Desktop/server archive. See `attention/README.md` for the compact-map rendering command.
