#!/usr/bin/env python3
"""Read-only source ingestion and independent semantic T3 assertion evaluation.

All generated files are written only below --out. Existing captions are never
regenerated, rewritten or relabelled in their source files. A model judgment is
made for every caption/annotated-negative-unit pair; no keyword verdicts exist.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import statistics
import time

TYPES = ['object', 'co_occurrence', 'attribute', 'relation']
GENERAL = {'InternVL3.5-8B', 'Qwen3-VL-8B', 'llava-ov-7b', 'qwen2.5-vl-7b'}
SOURCE = pathlib.Path('/home/u2025141034/benchmark/refcocog_eval_13models_4tasks_500/run_20260918_125802')
ANNOTATIONS = pathlib.Path('/home/u2025141034/benchmark/repaired/refcocog_500_dev.semantic_strict.json')
MODEL = '/home/u2025141034/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
SYSTEM = '''You are an independent semantic assertion annotator. Read only the supplied caption as evidence about what its author ASSERTS. Decide whether it asserts the edited negative semantic unit. The unit is a benchmark counterfactual; you are NOT checking the image, guessing its truth, or grading the caption's overall quality.

Rules:
1. Return 1 only for a definite assertion of the edited negative unit; otherwise return 0. Synonyms, faithful paraphrases and ordinary grammatical reformulations count. Mere token overlap does not.
2. object and co_occurrence: an introduced missing object asserted as present counts even if the caption omits the negative expression's unchanged positive contextual modifiers. However, if the annotation explicitly scopes the false object to a particular target or location and the same object exists elsewhere, a mention of that other object does not assert the scoped false unit. Use the annotation note only to understand the edited unit's scope, never as evidence of what the caption says.
3. attribute: the edited property must be assigned to the target entity. The same property assigned to a different object does not count. Generic mention of the target without the property does not count.
4. relation: the edited relation must bind the correct subject and reference object, with the correct direction and roles. Mentioning both objects, reversing the roles, or giving a different relation does not count. Equivalent inverse phrasing (for example, B is below A means A is above B) counts.
5. Negative statements, questions, hypotheticals, uncertainty (may, might, perhaps), quoted claims, and statements explicitly disclaimed by the caption do not count as definite assertions. Do not infer missing attributes, positions, entities, or relations from world knowledge.
6. Do not require the entire negative expression or all unchanged positive conditions. Judge the changed semantic unit, bound to its intended entity or roles.

Reply with exactly one character: 1 or 0. No explanation, punctuation, or other text.'''


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def judge_payload(rec):
    return {
        'caption': rec['caption'],
        'edit_type': rec['hallucination_type'],
        'positive_target_reference': rec['positive_text'],
        'edited_negative_expression': rec['negative_text'],
        'negative_head_object': rec['target_eval_units'].get('negative_head_object'),
        'edited_negative_semantic_units': rec['target_eval_units']['hallucination_units'],
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(ANNOTATIONS.read_text())
    groups = collections.defaultdict(dict)
    for a in annotations:
        typ = a['hallucination_type']
        assert typ in TYPES
        assert typ not in groups[a['base_sample_id']]
        units = a['chair_annotation']['target_eval_units']
        assert units.get('hallucination_units'), a['sample_id']
        assert units['eval_type'] == typ
        assert a['positive_text'].strip() and a['negative_text'].strip()
        groups[a['base_sample_id']][typ] = a
    assert len(groups) == 500
    assert all(set(v) == set(TYPES) for v in groups.values())
    assert len({a['image_filename'] for a in annotations}) == 500
    manifest = {'annotation_source': str(ANNOTATIONS), 'annotation_source_sha256': digest(ANNOTATIONS),
                'n_distinct_images': 500, 'n_base_groups': 500, 'n_annotation_pairs': len(annotations),
                'annotation_counts_by_type': dict(collections.Counter(a['hallucination_type'] for a in annotations)),
                'caption_sources': {}, 'metadata_review_status': 'Original pre-review flags preserved; final human-review status is supplied by the user and manuscript, not fabricated in these source rows.'}
    captions = []
    for path in sorted(SOURCE.glob('*/records.jsonl')):
        caps = [r for r in read_jsonl(path) if r.get('task') == 't3_pure_caption']
        if not caps:
            continue
        model = path.parent.name
        by = {r['base_sample_id']: r for r in caps}
        assert len(by) == len(caps) == 500, model
        assert set(by) == set(groups), model
        manifest['caption_sources'][model] = {'path': str(path), 'sha256': digest(path),
                                             'n_captions': len(caps), 'n_unique_bases': len(by)}
        for base_id in sorted(by):
            r = by[base_id]
            assert isinstance(r.get('caption'), str), (model, base_id)
            captions.append({'model': model, 'base_sample_id': base_id, 'caption': r['caption'],
                             'caption_sha256': hashlib.sha256(r['caption'].encode()).hexdigest(),
                             'source_sample_id': r['sample_id'], 'source_path': str(path),
                             'source_parse_failure': r.get('parse_failure'),
                             'original_query_role': r.get('query_role'),
                             'original_raw_output': r.get('raw_output')})
    assert len(manifest['caption_sources']) == 13
    assert len(captions) == 6500
    with open(out / 'cached_captions.jsonl', 'w') as f:
        for r in captions:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(out / 'source_annotations.jsonl', 'w') as f:
        for r in annotations:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(out / 'judgment_inputs.jsonl', 'w') as f:
        idx = 0
        for cap in captions:
            for typ in TYPES:
                a = groups[cap['base_sample_id']][typ]
                units = a['chair_annotation']['target_eval_units']
                r = {'index': idx, 'model': cap['model'], 'base_sample_id': cap['base_sample_id'],
                     'sample_id': a['sample_id'], 'pair_id': a['pair_id'],
                     'image_filename': a['image_filename'], 'hallucination_type': typ,
                     'semantic_group': 'EOH' if typ in TYPES[:2] else 'ROH',
                     'caption': cap['caption'], 'caption_sha256': cap['caption_sha256'],
                     'caption_source_path': cap['source_path'],
                     'positive_text': a['positive_text'], 'negative_text': a['negative_text'],
                     'target_eval_units': units, 'annotation_source_path': str(ANNOTATIONS)}
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                idx += 1
    manifest['n_real_cached_captions'] = len(captions)
    manifest['n_new_semantic_judgments'] = idx
    for name in ['cached_captions.jsonl', 'source_annotations.jsonl', 'judgment_inputs.jsonl']:
        manifest[name + '_sha256'] = digest(out / name)
    write_json(out / 'input_manifest.json', manifest)
    print(json.dumps({'prepared': True, 'n_captions': len(captions), 'n_judgments': idx}), flush=True)


def load_judge():
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                      attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    model.config.use_cache = False
    meta = {'judge_model': 'Qwen/Qwen3.5-9B', 'judge_snapshot': pathlib.Path(MODEL).name,
            'judge_path': MODEL, 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'dtype': 'bfloat16', 'attention': 'sdpa', 'device': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'generation': 'deterministic first-token argmax, retry unconstrained up to 16 tokens on nonbinary output',
            'enable_thinking': False, 'model_load_seconds': time.time() - start,
            'model_files': [{'name': p.name, 'target': str(p.resolve()),
                             'sha256': digest(p) if p.suffix != '.safetensors' else p.resolve().name}
                            for p in pathlib.Path(MODEL).iterdir() if p.is_file() and p.suffix in ['.json', '.safetensors']]}
    return tokenizer, model, meta


def classify(tokenizer, model, recs, torch):
    conversations = [[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user', 'content': json.dumps(judge_payload(r), ensure_ascii=False)}] for r in recs]
    texts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for x in conversations]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=False).to('cuda')
    binary_ids = [tokenizer.encode(str(v), add_special_tokens=False) for v in [0, 1]]
    assert all(len(x) == 1 for x in binary_ids), binary_ids
    with torch.inference_mode():
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
        tops = logits.argmax(dim=-1).tolist()
        scores = logits[:, [x[0] for x in binary_ids]]
        probs = scores.softmax(dim=-1)[:, 1].tolist()
    results = []
    for i, tok in enumerate(tops):
        raw = tokenizer.decode([tok]).strip()
        label = int(raw) if raw in ['0', '1'] else None
        retried = False
        if label is None:
            retried = True
            x = tokenizer(texts[i], return_tensors='pt', truncation=False).to('cuda')
            with torch.inference_mode():
                ids = model.generate(**x, max_new_tokens=16, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id)
            raw = tokenizer.decode(ids[0, x['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            label = int(raw) if raw in ['0', '1'] else None
        results.append({'asserted': bool(label) if label is not None else None,
                        'raw_judge_output': raw, 'judge_parse_valid': label is not None,
                        'p_asserted_conditional_binary': probs[i], 'retry_used': retried,
                        'input_token_count': int(inputs['attention_mask'][i].sum())})
    return results


def run(out, shard, nshards, batch, controls, limit):
    import torch
    path = out / ('controls.jsonl' if controls else 'judgment_inputs.jsonl')
    rows = [r for r in read_jsonl(path) if controls or r['index'] % nshards == shard]
    if limit:
        rows = rows[:limit]
    name = 'control_outputs.jsonl' if controls else ('pilot_outputs.jsonl' if limit else f'judgments_shard_{shard}.jsonl')
    dst = out / name
    done = {r['index'] for r in read_jsonl(dst)} if dst.exists() else set()
    rows = [r for r in rows if r['index'] not in done]
    tokenizer, model, meta = load_judge()
    write_json(out / ('judge_control_metadata.json' if controls else f'judge_metadata_shard_{shard}.json'), meta)
    start = time.time()
    with open(dst, 'a', buffering=1) as f:
        for off in range(0, len(rows), batch):
            chunk = rows[off:off + batch]
            pred = classify(tokenizer, model, chunk, torch)
            for row, p in zip(chunk, pred):
                rec = dict(row)
                rec.update(p)
                rec.update({'judge_model': meta['judge_model'], 'judge_snapshot': meta['judge_snapshot'],
                            'protocol_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
                            'shard': shard, 'nshards': nshards})
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if off == 0 or (off + batch) % (batch * 20) == 0 or off + batch >= len(rows):
                print(json.dumps({'shard': shard, 'done': min(off + batch, len(rows)), 'total': len(rows),
                                  'elapsed_seconds': time.time() - start,
                                  'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9}), flush=True)
    if controls:
        allrows = list(read_jsonl(dst))
        valid = [r for r in allrows if r['judge_parse_valid']]
        correct = sum(r['asserted'] == r['expected_asserted'] for r in valid)
        by = {}
        for cat in sorted({r['control_category'] for r in allrows}):
            rs = [r for r in allrows if r['control_category'] == cat]
            by[cat] = {'n': len(rs), 'correct': sum(r['judge_parse_valid'] and r['asserted'] == r['expected_asserted'] for r in rs)}
        report = {'n': len(allrows), 'n_valid': len(valid), 'correct': correct,
                  'accuracy_all_controls': correct / len(allrows), 'by_category': by,
                  'controls_are': 'Explicitly authored semantic diagnostic controls, not claimed to be human-labeled benchmark samples.',
                  'model_metadata': meta, 'failed_controls': [r for r in allrows if not r['judge_parse_valid'] or r['asserted'] != r['expected_asserted']]}
        write_json(out / 'control_summary.json', report)
        print(json.dumps({k: report[k] for k in ['n', 'n_valid', 'correct', 'accuracy_all_controls', 'by_category']}), flush=True)


def summarize(out, nshards):
    manifest = json.loads((out / 'input_manifest.json').read_text())
    rows = []
    for shard in range(nshards):
        rows.extend(read_jsonl(out / f'judgments_shard_{shard}.jsonl'))
    assert len(rows) == len({r['index'] for r in rows}) == 26000
    rows.sort(key=lambda r: r['index'])
    assert [r['index'] for r in rows] == list(range(26000))
    with open(out / 'judgments_all.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    per = {}
    for model in manifest['caption_sources']:
        rs = [r for r in rows if r['model'] == model]
        assert len(rs) == 2000
        by = {}
        for typ in TYPES:
            ss = [r for r in rs if r['hallucination_type'] == typ]
            assert len(ss) == 500
            valid = [r for r in ss if r['judge_parse_valid']]
            counts = sum(r['asserted'] for r in valid)
            by[typ] = {'n_units': len(ss), 'n_judge_valid': len(valid), 'n_judge_invalid': len(ss) - len(valid),
                       'assertion_count': counts, 'FAR': counts / len(ss) if len(valid) == len(ss) else None,
                       'FAR_valid_only': counts / len(valid) if valid else None}
        per[model] = {'family': 'general' if model in GENERAL else 'RL', 'n_captions': 500,
                      'n_images': 500, 'n_negative_units': 2000, 'by_type': by}
        for grp, types in [('EOH', TYPES[:2]), ('ROH', TYPES[2:])]:
            nn = sum(by[t]['n_units'] for t in types)
            invalid = sum(by[t]['n_judge_invalid'] for t in types)
            counts = sum(by[t]['assertion_count'] for t in types)
            per[model][grp] = {'n_units': nn, 'assertion_count': counts, 'n_judge_invalid': invalid,
                               'FAR': counts / nn if invalid == 0 else None,
                               'FAR_percent': 100 * counts / nn if invalid == 0 else None}
    families = {}
    for fam in ['general', 'RL']:
        ms = [m for m, p in per.items() if p['family'] == fam]
        families[fam] = {'n_models': len(ms), 'models': ms,
                         'EOH_FAR_percent': statistics.mean(per[m]['EOH']['FAR_percent'] for m in ms),
                         'ROH_FAR_percent': statistics.mean(per[m]['ROH']['FAR_percent'] for m in ms)}
    unchanged = digest(ANNOTATIONS) == manifest['annotation_source_sha256']
    unchanged = unchanged and all(digest(pathlib.Path(v['path'])) == v['sha256'] for v in manifest['caption_sources'].values())
    assert unchanged
    controls = json.loads((out / 'control_summary.json').read_text())
    summary = {'task': 'T3 query-free captioning', 'metric': 'caption false ASSERTION rate (FAR)',
               'caption_generation': 'Existing cached query-free outputs reused byte-for-byte; no query sent to the caption generation models.',
               'cohort': {'distinct_images_per_model': 500, 'captions_per_model': 500, 'n_models': 13,
                          'n_real_captions_total': 6500, 'negative_units_per_type_per_model': 500,
                          'EOH_units_per_model': 1000, 'ROH_units_per_model': 1000, 'n_judgments': 26000},
               'judge_model': controls['model_metadata'], 'control_accuracy': controls['accuracy_all_controls'],
               'control_n': controls['n'], 'per_model': per, 'family_equal_model_means': families,
               'protocol_system_prompt': SYSTEM, 'input_manifest_sha256': digest(out / 'input_manifest.json'),
               'judgments_all_sha256': digest(out / 'judgments_all.jsonl'), 'source_files_unchanged': unchanged,
               'all_judgments_parse_valid': all(r['judge_parse_valid'] for r in rows),
               'no_caption_cohort_doubling': True, 'original_legacy_object_scores_preserved': True}
    write_json(out / 't3_summary.json', summary)
    print(json.dumps({'per_model': per, 'family_means': families, 'all_valid': summary['all_judgments_parse_valid']}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'controls', 'run', 'summarize'])
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('/home/u2025141034/benchmark/rrwr_20261010_t3_attention/t3'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare(args.out)
    elif args.action == 'summarize':
        summarize(args.out, args.nshards)
    else:
        run(args.out, args.shard, args.nshards, args.batch, args.action == 'controls', args.limit)


if __name__ == '__main__':
    main()
