#!/usr/bin/env python3
"""OmniVerifier 作为 plug-and-play 过滤器，作用在 13 个上游模型 REAL T2 预测框上。

与 s5_grpo_filter.py 的关键差异（不可混用）：
  - JEV 吃「文本坐标」形式的框（训练时如此）。
  - OmniVerifier 吃「图上画红框」形式（我们 500-dev 验证时如此）。
  输入形式必须与各自训练/验证时一致，否则是分布外读数。

输出连续 logit z（不是二值决策），便于后续与 JEV 融合 + 阈值扫描。
每行: model, sid, htype, label_exists, iou, drew, z_omni, parsed
Resumable: append+flush，跳过已存在的 (model,sid)。
"""
import os, sys, json, argparse, time
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.expanduser('~/SVD/grpo_verifier'))

ROOT_11 = '/home/u2025141034/benchmark/refcocog_eval_11models_500_repaired/run_500_semantic_strict'
ROOT_13 = '/home/u2025141034/benchmark/refcocog_eval_13models_4tasks_500/run_20260918_125802'
IMG = '/home/u2025141034/models/LENS/data/refcoco/train2014'


def parse_bbox(v):
    if v in (None, 'None', ''):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def img_path(r):
    fn = r.get('image_filename') or r.get('image') or ''
    if fn:
        p = os.path.join(IMG, os.path.basename(fn))
        if os.path.exists(p):
            return p
    sid = r.get('base_sample_id') or r.get('sample_id', '')
    if 'COCO_train2014_' in sid:
        num = sid.split('COCO_train2014_')[-1].split('__')[0].split('_')[0]
        p = os.path.join(IMG, 'COCO_train2014_' + num + '.jpg')
        if os.path.exists(p):
            return p
    return None


def draw_box(path, bbox, max_side=896):
    img = Image.open(path).convert('RGB')
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w = max(2, int(round(min(img.size) * 0.008)))
    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=w)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
    return img


class Omni:
    def __init__(self, path, device):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.device = device
        tok = self.proc.tokenizer
        t, f = [], []
        for s in ('true', ' true', 'True', ' True'):
            t += tok.encode(s, add_special_tokens=False)[:1]
        for s in ('false', ' false', 'False', ' False'):
            f += tok.encode(s, add_special_tokens=False)[:1]
        self.ids_true, self.ids_false = sorted(set(t)), sorted(set(f))

    def question(self, phrase):
        return (
            'The red rectangle in the image marks a candidate region. '
            f'Claim about the red-boxed region: "{phrase}".\n'
            'Carefully analyze the image and determine whether the object, its attributes, '
            'and its spatial relationships stated in the claim are all correctly satisfied '
            'by the red-boxed region. If the claim is accurate, answer true; otherwise false.\n'
            'Respond strictly in JSON: {"answer": true/false}'
        )

    @torch.no_grad()
    def score(self, image_path, bbox, phrase):
        img = draw_box(image_path, bbox)
        msgs = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': self.question(phrase)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = text + '{"answer":'
        inputs = self.proc(text=[text], images=[img], return_tensors='pt').to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        zt = torch.logsumexp(logits[self.ids_true], 0).item()
        zf = torch.logsumexp(logits[self.ids_false], 0).item()
        return zt - zf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--omni_path', default=os.path.expanduser('~/models/OmniVerifier-7B'))
    ap.add_argument('--tag', default='omni')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--outdir', default=os.path.expanduser('~/SVD/grpo_verifier/s5grpo'))
    ap.add_argument('--root', default='13')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    root = ROOT_13 if args.root == '13' else (ROOT_11 if args.root in ('', '11') else args.root)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.outdir, exist_ok=True)

    model = Omni(args.omni_path, 'cuda:0')

    for mname in args.models.split(','):
        mname = mname.strip()
        rec_path = os.path.join(root, mname, 'records.jsonl')
        if not os.path.exists(rec_path):
            print(f'[skip] {mname}: no records.jsonl', flush=True)
            continue
        out_path = os.path.join(args.outdir, f's5omni_{args.tag}_{mname}.jsonl')
        done = set()
        if os.path.exists(out_path):
            for l in open(out_path, encoding='utf-8'):
                try:
                    done.add(json.loads(l)['sid'])
                except Exception:
                    pass
        out = open(out_path, 'a', encoding='utf-8')

        rows = []
        for l in open(rec_path, encoding='utf-8'):
            try:
                r = json.loads(l)
            except Exception:
                continue
            task = str(r.get('task', '')).lower()
            if task not in ('t2', 't2_vqa_grounding'):
                continue
            rows.append(r)
        if args.limit:
            rows = rows[:args.limit]

        t0, n = time.time(), 0
        for r in rows:
            sid = r.get('sample_id') or r.get('base_sample_id')
            if sid in done:
                continue
            bb = parse_bbox(r.get('pred_bbox_xyxy'))
            drew = bb is not None
            rec = dict(model=mname, sid=sid,
                       htype=r.get('hallucination_type') or r.get('query_role'),
                       label_exists=bool(r.get('label_exists')),
                       iou=float(r.get('iou') or 0.0), drew=drew, z_omni=None, parsed=False)
            if drew:
                ip = img_path(r)
                phrase = r.get('query') or r.get('referring_expression') or ''
                if ip and phrase:
                    try:
                        rec['z_omni'] = model.score(ip, bb, phrase)
                        rec['parsed'] = True
                    except Exception as exc:
                        rec['error'] = f'{type(exc).__name__}: {exc}'
            out.write(json.dumps(rec) + '\n')
            out.flush()
            n += 1
            if n % 200 == 0:
                print(f'  {mname} {n}/{len(rows)} {(time.time()-t0)/n:.2f}s/it', flush=True)
        out.close()
        print(f'DONE {mname} n={n} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
