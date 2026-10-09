"""Verify or materialize the published archive without changing the locked scorer."""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPES = ('object', 'co_occurrence', 'attribute', 'relation')

def records():
    with gzip.open(HERE / 'judgments_all.jsonl.gz', 'rt') as f:
        for line in f:
            yield json.loads(line)

def verify():
    counts = collections.defaultdict(lambda: collections.Counter())
    images = collections.defaultdict(set)
    identities = set()
    indices = set()
    n = 0
    for r in records():
        assert r['judge_parse_valid'], r['index']
        assert hashlib.sha256(r['caption'].encode()).hexdigest() == r['caption_sha256']
        key = (r['model'], r['base_sample_id'], r['hallucination_type'])
        assert key not in identities and r['index'] not in indices
        identities.add(key); indices.add(r['index'])
        model, typ = r['model'], r['hallucination_type']
        counts[model][typ + '_n'] += 1
        counts[model][typ + '_asserted'] += int(r['asserted'])
        images[model].add(r['base_sample_id'])
        n += 1
    assert n == 26000 and len(counts) == 13 and len(indices) == 26000
    source = json.loads((HERE / 't3_summary.json').read_text())
    for model, row in counts.items():
        assert len(images[model]) == 500
        for typ in TYPES:
            assert row[typ + '_n'] == 500
            assert row[typ + '_asserted'] == source['per_model'][model]['by_type'][typ]['assertion_count']
        for group, types in [('EOH', TYPES[:2]), ('ROH', TYPES[2:])]:
            total = sum(row[t + '_asserted'] for t in types)
            assert total == source['per_model'][model][group]['assertion_count']
            assert source['per_model'][model][group]['n_units'] == 1000
    return {'records': n, 'models': len(counts), 'captions_per_model': 500,
            'negative_units_per_type_per_model': 500, 'summary_counts_verified': True,
            'caption_sha_verified': True, 'unique_record_ids_verified': True}

def materialize(dest, shards=False):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['judgment_inputs', 'judgments_all', 'cached_captions', 'source_annotations']:
        output = dest / (name + '.jsonl')
        with gzip.open(HERE / (name + '.jsonl.gz'), 'rb') as f:
            data = f.read()
        if output.exists():
            assert output.read_bytes() == data, f'Existing output differs: {output}'
        else:
            output.write_bytes(data)
    for name in ['input_manifest.json', 'controls.jsonl', 'control_final_summary.json',
                 'heldout_controls_independent.jsonl', 'heldout_summary.json']:
        output = dest / name
        data = (HERE / name).read_bytes()
        if output.exists():
            assert output.read_bytes() == data, f'Existing output differs: {output}'
        else:
            output.write_bytes(data)
    if shards:
        paths = [dest / f'judgments_shard_{i}.jsonl' for i in range(4)]
        assert not any(p.exists() for p in paths), 'Choose a fresh directory for shard materialization.'
        handles = [p.open('w') for p in paths]
        try:
            for r in records():
                handles[r['index'] % 4].write(json.dumps(r, ensure_ascii=False) + '\n')
        finally:
            for f in handles:
                f.close()

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=['verify', 'materialize', 'rescore'])
    cli.add_argument('--out', type=Path)
    cli.add_argument('--restore-shards', action='store_true')
    cli.add_argument('--model-path', type=Path)
    cli.add_argument('--shard', type=int, default=0)
    cli.add_argument('--nshards', type=int, default=4)
    cli.add_argument('--batch', type=int, default=8)
    args = cli.parse_args()
    if args.action == 'verify':
        print(json.dumps(verify()))
        return
    if args.out is None:
        cli.error('--out is required; use a new replay directory')
    if args.action == 'materialize':
        materialize(args.out, args.restore_shards)
        print(json.dumps({'out': str(args.out), 'restored_shards': args.restore_shards}))
        return
    if args.model_path is None:
        cli.error('--model-path is required for rescore')
    assert (args.out / 'judgment_inputs.jsonl').is_file(), 'Materialize archive inputs first.'
    lock = json.loads((HERE / 'protocol_lock.json').read_text())
    assert hashlib.sha256((HERE / 't3_semantic_eval.py').read_bytes()).hexdigest() == lock['locked_script_sha256']
    held = json.loads((HERE / 'heldout_summary.json').read_text())
    assert held['n'] == 64 and held['accuracy_all_controls'] >= .95
    spec = importlib.util.spec_from_file_location('locked_semantic_scorer', HERE / 't3_semantic_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL = str(args.model_path.resolve())
    module.run(args.out, args.shard, args.nshards, args.batch, False, 0)

if __name__ == '__main__':
    main()
