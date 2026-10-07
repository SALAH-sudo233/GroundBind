# GroundBind 补充实验完整协议

本文件详述三个新增消融实验的完整操作定义，回应审稿人对"实验协议不够清晰"的关切。所有数字可由本目录脚本在 vlm1 (`~/SVD/agentic_probe`) 复算。

---

## 1. 图像交换消融（Wrong-Image Control）

**目的**：验证验证器的额外调用是否真正使用**当前探针图像**，还是仅依赖文本先验。

### 样本筛选
- **基准集**：`probe_upstream.jsonl` 中所有触发结构分支（relation 幻觉）的候选
- **前提条件**：候选框 `pred_bbox_xyxy` 存在且可解析（4 元素数组或 JSON 字符串）
- **图像可用性**：能从 `sample_id` 提取 COCO 文件名（格式 `hallu_*_COCO_train2014_*.jpg`）且文件存在于 `~/models/LENS/data/refcoco/train2014/`

### 图像替换方式
**标签盲、确定性、尺寸安全**：
1. **候选池**：本子集所有图像的排序集合（按文件名字典序）
2. **索引函数**：`md5(model|sample_id) % N`，从该位置向前遍历
3. **选择条件**：
   - 错图文件名 ≠ 原图文件名
   - 原框坐标 `[x0, y0, x1, y1]` 完全落在错图真实宽高内（`x1 ≤ W_wrong, y1 ≤ H_wrong`）
   - 若无满足条件的错图，该行**跳过**并计入 `n_skip`（不修改尺寸以凑出结果）
4. **复现性**：相同 model+sample_id 总映射到同一错图（md5 seed 固定）

### 提示词
**原图与错图使用完全相同的文本与框坐标**：
```
The red rectangle in the image marks a candidate region. 
Claim about the red-boxed region: "{phrase}".
Carefully analyze the image and determine whether the object, its 
attributes, and its spatial relationships stated in the claim are 
all correctly satisfied by the red-boxed region. 
If the claim is accurate, answer true; otherwise false.
Respond strictly in JSON: {"answer": true/false}
```
- **画框方式**：PIL `ImageDraw.rectangle`，红色 `(255,0,0)`，线宽 `max(2, round(min(W,H)*0.008))`
- **图像预处理**：画框后，若 `max(W,H) > 896`，等比例缩放至长边 896（BICUBIC）
- **错图提示**：框坐标、phrase、提示词模板与原图**完全相同**，仅图像像素不同

### 差值定义
**z = OmniVerifier 的 true-vs-false logit 差**（越高越"接受"）：
```python
logits = model(**inputs).logits[0, -1].float()
ids_true  = tokenizer.encode(['true', ' true', 'True', ' True'], add_special_tokens=False)[:1]  # 去重
ids_false = tokenizer.encode(['false', ' false', 'False', ' False'], add_special_tokens=False)[:1]
z = logsumexp(logits[ids_true]) - logsumexp(logits[ids_false])
```
- **delta 行级**：`z_real - z_swap`（正数 = 原图分数更高）
- **delta 模型级**：该模型所有行的 delta 算术平均

### 聚合与不确定性
**模型级配对 bootstrap**（13 个上游模型等权）：
1. 从 13 个模型的 delta 均值中有放回抽样 13 次
2. 计算该样本的总均值
3. 重复 2000 次，取 2.5% / 97.5% 分位数为 95% CI
4. **显著性**：CI 不跨零

**报告字段**（`attribution.json` → `stats.wrong_image`）：
- `n_rows`：有效行数（完成原图+错图双采）
- `mean_real` / `mean_swap`：汇总 z 均值
- `delta`：模型级 delta 均值
- `ci`：[下界, 上界]
- `significant`：布尔

**实现**：`collect_aswap.py`（采集）+ `eval_attribution.py` L316-338（统计）

---

## 2. 去框消融（No-Box Ablation）

**目的**：验证 OmniVerifier 的判别力是否依赖**候选框的存在**（region-localized）还是仅需图像+表达一致性（image-grounded）。

### 样本筛选
**复用图像交换消融的同一候选集**（保证两消融可比）：
- 基准：`probe_upstream.jsonl` 触发结构分支的候选
- 前提：`pred_bbox_xyxy` 存在、图像可读、query 文本（来自 `probe_textroute_all.jsonl` 对应 model+sid 的 `variants[*].query`，取第一个非空）

### 提示词
**有框（boxed）**：同图像交换消融的红框提示（见上节）

**无框（no-box）**：
```
Claim about the image: "{phrase}".
Carefully analyze the image and determine whether the object, its 
attributes, and its spatial relationships stated in the claim are 
all correctly satisfied by the image. 
If the claim is accurate, answer true; otherwise false.
Respond strictly in JSON: {"answer": true/false}
```
- **区别**：不画红框，提示词改"Claim about **the image**"（不提"region"/"red-boxed"）
- **图像预处理**：原图直接缩放（若 `max(W,H) > 896`），无框绘制

### 差值定义
**z 计算口径同上**（true-vs-false logit 差）

**delta 行级**：`z_nobox - z_box`
- 正数 = 去框后分数更高（违背"需要框"假设）
- 负数 = 有框时分数更高（符合"需要框"）

### 聚合与不确定性
**AUROC 对比**（Mann-Whitney）：
- **label**：从 `probe_textroute_all.jsonl` 的 `htype` 字段读取
  - `htype == 'positive'` → label=1
  - 其余（object/co_occurrence/attribute/relation 负例） → label=0
  - 若 htype 缺失，从 `sample_id` 后缀判断（无 `__` = positive）
- **AUROC(有框)**：用 `z_box` 分离 positive/非positive 的能力
- **AUROC(无框)**：用 `z_nobox` 分离能力
- **degraded 模型数**：`AUROC(无框) < AUROC(有框)` 的模型计数（13 模型中）

**报告字段**（`nobox.json`）：
- `n` / `n_pos` / `n_neg`：总行数 / positive / 其余
- `dz_mean` / `dz_median`：delta 均值/中位数
- `auroc_box` / `auroc_nobox`：汇总 AUROC
- `auroc_drop`：有框 - 无框
- `per_model`：逐模型统计（含 AUROC / drop / n_pos / n_neg）
- `models_degraded`：去框后 AUROC 下降的模型数
- `n_models`：13

**实现**：`collect_nobox.py`（8 卡并行，11909 行）+ `eval_nobox.py`（合并 shard + 统计）

---

## 3. 谓词去均值消融（Predicate De-meaning）

**目的**：检验 relation gap（竞争探针分数 - support 探针分数）的判别力是否**冗余于谓词身份**（可从谓词类型预测 gap）还是独立信息源。

### 样本筛选
- **基准集**：`probe_upstream.jsonl` + `trprobe_jev_all.jsonl` 合并后 `drew=True` 且 `z_o/z_j` 非 None 的行
- **谓词提取**：从 query 文本正则匹配 48 个空间关系谓词（`simple_relations.py` 的 `ALL_RELS`）
- **gap 定义**：
  ```python
  g_o = zmax_o - z_o  # OmniVerifier 竞争 gap
  g_j = zmax_j - z_j  # JEV-2B 竞争 gap
  ```
  其中 `zmax_*` 是该候选所有竞争探针的最高分（来自 `probe_textroute_all.jsonl` 的 `variants` 字段）

### 去均值操作
**一阶方差分解（单因素 ANOVA R²）**：
1. 按谓词分组，计算每个谓词的 gap 均值 `μ_pred`
2. **组间平方和**：`SSB = Σ_pred n_pred * (μ_pred - μ_global)²`
3. **总平方和**：`SST = Σ_i (gap_i - μ_global)²`
4. **R² = SSB / SST**（谓词身份解释的 gap 方差比例）
5. **residual gap = gap - μ_pred**（去除谓词均值后的残差）

### 判别力指标
**AUROC（Mann-Whitney）**：
- **label**：从 `sample_id` 后缀判断（无 `__` = positive → 1，有 `__` = 负例 → 0）
- **AUROC(gap)**：用原始 gap 分离 positive/负例
- **AUROC(residual)**：用去均值后的 residual 分离
- **塌陷量**：`AUROC(gap) - AUROC(residual)`
  - 接近 0 = residual 保留判别力（gap 不冗余于谓词）
  - 显著 > 0 = residual 塌陷（gap 可从谓词预测）

### 聚合与不确定性
**模型级统计**（13 模型）：
- 每个模型独立算 R² / AUROC(gap) / AUROC(residual)
- 汇总报告：R² 均值、AUROC 均值、塌陷量的配对 bootstrap CI（同图像交换的方法，2000 次重采样）

**组内 AUROC**（predicate identity 常数）：
- 每个谓词内部单独算 AUROC（此时谓词身份固定，prior=0）
- 13 模型 × 48 谓词 → 报告均值/中位数/有效谓词数

**报告字段**（`gap_vs_prior.json`）：
- `per_model[model]`：`n` / `r2_predicate` / `auroc_gap` / `auroc_residual` / `within_predicate_auroc_stats`
- 汇总：pooled R² / AUROC 均值 / residual-raw 配对差的 CI

**实现**：`diag_gap_vs_prior.py`

---

## 数据隔离与复现性

### 图像根
所有实验图像路径：`~/models/LENS/data/refcoco/train2014/`（COCO 2014 train split，与 benchmark 构造同源）

### 模型
- **OmniVerifier**：`Qwen/Qwen2.5-VL-7B-Instruct`（本地路径 `~/models/LENS/Qwen2.5-VL-7B-Instruct/`）
- **推理参数**：bfloat16, `sdpa` attention, greedy decoding（temperature=0 等效，取 `logits[0, -1]` 直接算 z）
- **max_side=896**：长边缩放上限（保持宽高比）

### Seed 与确定性
- **图像交换**：md5(model|sample_id) 决定错图索引（确定性，标签盲）
- **谓词去均值**：无随机性（分组统计）
- **bootstrap CI**：`seed=0` 固定（`random.Random(0)`）

### 产物文件
所有结果 JSON 与脚本已提交至 `reviewer_v060/`：
- `swap_all.jsonl`（13562 行，图像交换原始采集）
- `nobox_all.jsonl`（11909 行，去框采集）
- `attribution.json`（汇总统计，含 wrong_image / equal_budget / text_prior 对比）
- `nobox.json`（AUROC 对比）
- `gap_vs_prior.json`（R² / residual AUROC）

**复现命令**（vlm1 `~/SVD/agentic_probe`）：
```bash
# 图像交换采集（8 卡并行）
for s in {0..7}; do
  CUDA_VISIBLE_DEVICES=$s python collect_aswap.py --out swap_shard$s.jsonl --shard $s --nshards 8 &
done
cat swap_shard*.jsonl > swap_all.jsonl

# 去框采集（8 卡并行）
for s in {0..7}; do
  CUDA_VISIBLE_DEVICES=$s python collect_nobox.py --shard $s --nshards 8 &
done
cat nobox_shard*.jsonl > nobox_all.jsonl

# 统计
python eval_attribution.py --json-out attribution.json
python eval_nobox.py
python diag_gap_vs_prior.py
```

---

## 与论文对齐

**论文引用的数字**（截至提交版）：
- §5.2 末尾：图像交换 logit shift **Δ=11.41** [10.88, 11.84]（显著）
- Appendix 消融章节：去框 AUROC 0.5923→**0.6294**（0/13 模型下降）
- Q1 补充实验：R²(predicate) 均值 **0.080**，AUROC(gap) 0.613 vs AUROC(residual) 0.598

**本协议文件作为 supporting material 随仓库提交**，论文正文引用本文件 §1-3 对应章节以满足"实验定义需与结果同等充分写进论文"的要求。
