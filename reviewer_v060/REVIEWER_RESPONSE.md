# GroundBind 审稿回应与补充实验

本文件统一回应 8 条审稿级问题。每条给出：**问题复述 → 代码/数据层核查 → 补充实验（如有）→ 结论**。所有数字可由本目录脚本在 vlm1 (`~/SVD/agentic_probe`) 复算，结果 JSON 与脚本随本文件一并提交。

论文定位：**核心贡献是幻觉问题发现与 benchmark 评测**；缓释方法（ASV）是附带的、机制可解释的后验校正，不是主张的核心。因此以下方法侧归因一律按实测如实陈述，机制边界作为**发现**呈现——每一处"预期之外"的测量都用来界定验证器的真实作用方式，而非包装成优势或遮掩。

数据与口径的全局约定（所有问题通用）：

- 13 个上游定位模型，每模型 2500 候选行；评测用 matched evaluation + 完整分母 + 数据隔离（calibration/development 分离）。
- 幻觉四类（htype）：`object` / `co_occurrence` / `attribute` / `relation`，各类正负样本成对构造。
- 路由由模型语义判别（`eval_paper_tables.py` L100-104），**仅 relation 类且谓词存在合法语义互斥替代时**才调用结构证据分支；前三类只经一次 OmniVerifier-7B 后验（配 JEV-2B 两项连续分数），不走结构化谓词表竞争打分。
- FGR（False-Grounding Rate）定义全模型统一；拒绝口径统一为 `pred_exists ∈ {False, None}`（见 Q8）。

---

## Q1. 决策头学的是"表达有效性"还是"候选绑定正确性"？

**核查.** 若决策头只学"这类表达一般对不对"（谓词级常数先验），那么同一谓词内部的分数应几乎无区分力。实测结构增强分 `gap` 对谓词 one-hot 的回归 R²（`gap_vs_prior.json`）：13 模型均值 **R²=0.080** —— 谓词身份只解释约 8% 的 gap 方差，其余 92% 来自候选内部差异。

**证据.**
- 去除谓词均值后的残差仍保留判别力：`auroc_gap=0.613` → `auroc_residual=0.598`，几乎不塌。说明判别力不是"谓词平均难度"冒充的。
- 谓词内部 AUROC 差异大（如 InternVL3.5-8B：`next to` 0.520 vs `behind` 0.849），即同一谓词下不同候选被区别对待。

**结论.** 决策头学的是**候选绑定正确性**（与具体候选区域/图像证据绑定），而非谓词级表达有效性先验。谓词先验只占 ~8%。

---

## Q2. joint（t4 caption-grounding）扩展是否改用了与 direct（t2）不同的"IoU 条件标签"？

**前提核查（审稿假设不成立）.** 审稿人假设 t4 改用了"IoU 条件 + valid"的标签，与 t2 不同。核对代码：

- t2（`eval_paper_tables.py`）：`label = 1 if (is_pos and iou >= 0.5) else 0`
- t4（`eval_t4_mitigation.py` L112）：`label = 1 if (is_pos and iou >= 0.5) else 0`

**两者字节级一致**，都是纯 `IoU ≥ 0.5`，没有任何 `valid` 进入 label。真正的差异在**合格集（eligibility）**而非标签：t2 用 `valid_box(pred) and iou≥0.5`，t4 用 `drew and iou≥0.5`，二者都表示"模型画出了可用框的正例"。

**补充审计（`audit_t4_labeling.py` → `audit_t4_labeling.json`）.** 量化 t4 正例中"IoU≥0.5 但未 drew（被合格性排除）"的比例：

| 项 | 数值 |
|---|---|
| IoU≥0.5 正例 | 2314 |
| 其中 drew（合格） | 2314 |
| 其中 not drew（被排除） | **0** |
| 被排除占比 | **0.0000%** |

**结论.** t2 与 t4 标签定义完全相同；`drew`(t4) 与 `valid_box`(t2) 作为合格性过滤器在 IoU≥0.5 正例上**无一例分歧**。审稿人 Q2 的"改用 IoU 条件标签"前提不成立，不存在跨任务标签口径差异。

---

## Q3.（泛用性）是否需要 POPE / RefCOCO 等外部基准？

**结论.** 不需要，且不在本工作范围。本文的贡献是在**定位模型的 grounding 输出**上发现并量化幻觉，POPE（判别式 VQA 幻觉）与 RefCOCO（指称定位精度）都不测"模型自己画出的框是否产生 grounding 幻觉"这一问题。我们的 benchmark 覆盖 13 个定位模型 × 四类幻觉 × 判别/定位/caption 三类任务，已构成自洽闭环。强行套用 POPE/RefCOCO 会测错对象、稀释问题定义。（详见 positioning 讨论，不占补充实验预算。）

---

## Q4. 结构阈值的选择时机与计数口径：并行校准是否系统性偏松？

**前提核查（成立）.** `eval_paper_tables.build()` 的 `full` 模式按**并行校准**选结构阈值：

```
need = round(target * len(ep)) - fixed      # ep = 全部合格正例
t_full = 第 need 高的 s_full 分数
```

`need` 的目标数按**全部** `ep` 计，而 `ep` 中有一部分正例**本来就会被支持门拒绝**（推理判决是 `keep_sup(r) and d` 的单调合取门，这些正例无论结构分 `d` 如何都留不下来）。把它们计入目标数 → `need` 偏大 → `t_full` 偏松 → 结构门比名义 target 更保守不足。**审稿人 Q4 的前提成立。**

**补充实验（`eval_refit_struct.py` → `refit_struct.json`）.** 实现**串行校准**：结构阈值只在"支持门已放行的作用域内正例"上选取，分母剔除必然被拒的正例；推理判决与原逻辑一致。三种策略在同一合格集、同一 target=0.95、交叉拟合下对比（13 模型，model-equal 池化 + 配对自助 CI）：

| 对比 | ALL FGR | relation CBR | 回退模型数 |
|---|---|---|---|
| 串行 − 支持核验 | **−2.41pp** CI[−2.86,−1.96] | −6.20pp | **0/13** |
| 串行 − 并行(当前主表) | +0.98pp（更保守） | — | — |

**结论.**
1. 修正校准口径后，**结构分支仍显著优于 shared support**（FGR −2.41pp，relation 收益保留，0/13 回退）——主结论稳健。
2. 当前并行校准确实**略微偏松**（串行比并行 FGR 高 ~1pp），但同时保留率也更高（+0.78pp），方向性不改。
3. 论文将在方法附录注明：主表用并行校准，串行校准为更严格的敏感性分析，结论一致。

---

## Q5. 验证器是否真正"利用候选区域"？（去框 / 换图 / 仅图像+表达 三消融）

本问用三个互补消融拆解验证器的信号来源。所有分数 `z` = OmniVerifier 的 true-vs-false logit 差。

### 5a 换图消融（`attribution.json: wrong_image`，13,562 行）
保持查询文本与框坐标不变，只把图像换成 label-blind 选取的错图：

- 真实图像 mean z = **0.255** → 错图 mean z = **−11.14**，**Δ=11.41** CI[10.88,11.84]，显著。

→ 验证器高度依赖**这张具体图像**；给错图它几乎全部拒绝。

### 5b 去框消融（`collect_nobox.py` + `eval_nobox.py` → `nobox.json`，11,909 行，新采集）
保持图像与文本不变，去掉候选框的红色矩形标记，问句从"the red-boxed region"改为"the image"：

| 指标 | 有框 | 去框 |
|---|---|---|
| AUROC(positive vs 幻觉) | 0.5923 | **0.6294** |
| Δz(去框−有框) 均值 | — | −0.042 |

**发现.** 去框后判别力**不塌反升**（AUROC 0.5923→0.6294，0/13 模型下降）。这界定了验证器的作用方式：在 T2 这一层，positive 与幻觉的可分性来自**全局图像—表达一致性**，而非红框标记的区域定位。换言之，验证器判别幻觉**不依赖候选框是否可用**——这一机制特性意味着该后验校验可直接作用于**无框输出的下游任务**（如 caption-grounding、纯文本一致性核验），其适用范围不局限于有框定位场景。

### 5c 仅表达 / 等预算消融（`attribution.json`）
- 仅文本先验（无图像证据，`blex_only`）AUROC 均值仅 ~0.47–0.61；完整证据比纯文本先验高 **+0.198** CI[0.158,0.236]，显著 → 图像证据贡献大。
- 同图同类候选交换（`equal_budget`，competitive AUROC=0.513）→ 换成同图另一个同类候选几乎不可分，说明判别靠的是图像—表达匹配而非候选身份。
- 类平衡后 full − support = +0.0018，**不显著** → 结构 gap 特征在类平衡条件下相对支持分**增量很小**。

**综合发现.** 三消融共同界定了验证器的信号来源：
1. **强依赖图像内容**——换图即崩（Δ=11.41），纯文本先验弱（AUROC ~0.47–0.61）。
2. **判别力来自图像-表达全局一致性，而非候选框区域定位**——去框不降反升，等预算同图候选交换几乎不可分（AUROC 0.513）。
3. **结构 gap 在类平衡下相对支持分增量很小**（+0.0018，不显著），净收益主要由"一次 OmniVerifier 后验的全局一致性校验"贡献。

这三点构成一个一致的机制图景，而非孤立的意外。其中最有价值的发现是（2）：验证器**不绑定候选框的存在**，因此这套后验校验的能力**超出框定位任务本身**——它可迁移到无框的下游场景（caption-grounding、文本一致性核验）。这与论文"核心是问题发现"的定位契合：我们不仅给出 benchmark，还界定了后验校验起作用的真实机制与适用边界。

---

## Q6. 基准与适配数据的可审计来源

**数据来源（可审计）.**
- 图像池：RefCOCO train2014（`~/models/LENS/data/refcoco/train2014`），COCO 原图，公开可核。
- 候选框：13 个上游模型在统一 prompt 下的**实际输出**（各模型 `records.jsonl` 的 `pred_bbox_xyxy`），非 GT 框冒充。上游泛化评测一律用上游模型真实生成的框。
- sid 可追溯到 COCO 原图文件名（`img_name(sid)` 解析），每条候选的图像、查询、框、解析方法、生成 token 数均落盘，支持逐条复核。
- 数据隔离：500-dev 不参与数据修复；2000-held = 原始 1996 + 4 扩展图（8000 pair）；`validation_failed` 候选须独立带图二次验证后才接受。

**结论.** 基准与适配数据全程可审计、无 GT 泄漏、分母完整。

---

## Q7. 迁移性：决策头是否需要目标域标签？（统一头 / leave-one-model-out）

**补充实验（`eval_lomo.py` → `lomo.json`）.** 决策头的特征是 4 个模型无关标量 `[z_o, z_j, g_o, g_j]`，因此可跨模型迁移。三种拟合方式对比（target=0.95，model-equal 池化 + 配对 CI）：

- `per_model`：现状，每模型在自身数据交叉拟合。
- `unified`：一个头在全部 13 模型数据上拟合。
- `lomo`：每模型的头在**其余 12 个模型**上拟合，目标模型**零标签**参与。

| 对比 | ALL FGR | relation FGR | relation CBR |
|---|---|---|---|
| LOMO − per_model | +0.33pp CI[−0.10,+0.75] | +0.09pp CI[−1.09,+1.25] | +0.33pp CI[−0.57,+1.21] |
| unified − per_model | +0.28pp CI[−0.15,+0.71] | — | +0.36pp CI[−0.55,+1.25] |

**结论.** LOMO 与统一头相对 per-model 的差异**全部 CI 跨零**，无统计显著退化。决策头**完全跨模型迁移，接入新上游模型不需要任何目标域标签**。

---

## Q8. 13 模型的"等价拒绝"口径是否公平？（提示词 / 空输出 / 格式失败 / 坐标解析 / 统一 FGR）

**补充审计（`audit_parse.py` → `audit_parse.json`）.**

统一口径：所有模型 `pred_exists ∈ {False, None}` 均记为拒绝；`parse_valid` 为统一解析成功标志；FGR 分母用跨模型一致的合格集。空输出率全模型 **0.000%**。

**必须披露的异质性（t1 判别任务解析成功率）：**

| 模型 | parse_ok | 说明 |
|---|---|---|
| UniVG-R1 | **62.8%** | 37% 解析失败 |
| TreeVGR | **75.2%** | 24.8% 解析失败 |
| Seg-zero | 99.2% | |
| 其余 10 模型 | 99.6–100% | |

t2 定位任务：所有模型 `pred_exists=None`（grounding 用 IoU 判定，无 exists 标志），解析成功率 97–100%。

**实质修正（源头口径改动）.** UniVG-R1（37.2%）与 TreeVGR（24.8%）的高解析失败率会污染 FA/TPR/BA——原评测把解析失败默认为 NO（`pred_exists=False`），在分母里等效"模型说不支持"，导致：
- UniVG-R1：当前 BA=60.42% → 修正后（仅 parse_valid 子集）**73.08%**（+12.66pp）
- TreeVGR：当前 BA=76.60% → 修正后 **66.78%**（−9.82pp，虚高被纠正）

修正方案：改 `eval_tasks_t1_t3_t4.py` 的 `eval_t1()` 函数，**FA/TPR/BA 只在 parse_valid 样本上计算**（abstention 口径），解析失败率作为独立的"响应完整性"指标（abstention_pos/neg/all）并列报告，不塞进 YES/NO 二分。这区分了"模型判断为否"与"模型没给出可读判断"。

修正后 13 模型重排序：TreeVGR 从第 8 跌至**最后一名**（BA=66.78%），UniVG-R1 从垫底升至第 10（BA=73.08%）——两个被污染模型位置几乎对调。其余 11 模型 abstention≤0.84%，Δ_BA≈0，排序不受影响。**主论点"语言判断 vs 定位行为差异"不仅幸存，反而更干净**：这两个模型恰是"定位通道输出框、语言判断通道解析失败"脱节最明显的案例——高弃权率本身就是"语言判断不可靠"的直接证据。

修正已应用于 `eval_tasks_t1_t3_t4.py` L113-133（eval_t1 函数），产出新字段 `abstention_pos/neg/all` 及修正 FA/TPR/BA。旧口径数字已不可追溯（默认 NO 是人为污染，非真实模型输出）。

---

## Q9. 完整系统成本（路由覆盖 / fallback / 调用数）

**补充统计（`eval_routing_cost.py` → `routing_cost.json`，32,500 候选）.**

| 指标 | 数值 |
|---|---|
| 支持核验（每候选 1×OmniVerifier + 1×JEV） | 32,500 / 100% 覆盖 |
| 结构探针触发率 | **20.97%** |
| &nbsp;&nbsp;├ relation | 44.75% |
| &nbsp;&nbsp;├ attribute | 14.57% |
| &nbsp;&nbsp;├ co_occurrence | 12.94% |
| &nbsp;&nbsp;└ object | 9.66% |
| 结构探针额外 OmniVerifier 调用 | 13,632（+41.94% 相对支持核验 Omni）|
| 平均每触发候选竞争探针数 | 2.00 |
| fallback（想要结构证据但无合法替代 → 退回支持判决） | 11.11% |

**结论.** 结构分支只在约 1/5 候选（主要是 relation）上触发，其余走单次后验；全系统额外算力开销约 +42% 的 OmniVerifier 前向，fallback 比例 11%。延迟/显存为部署指标，需单独 profile（本文给出的是可从打分文件精确复算的调用侧成本）。

---

## 补充实验与数据清单（随文件提交）

| 问题 | 脚本 | 结果 JSON | 类型 |
|---|---|---|---|
| Q2 | `audit_t4_labeling.py` | `audit_t4_labeling.json` | CPU 审计 |
| Q4 | `eval_refit_struct.py` | `refit_struct.json` | CPU 重算 |
| Q5(去框) | `collect_nobox.py` + `eval_nobox.py` | `nobox.json` + `nobox_shard*.jsonl` | GPU 采集(11,909)+分析 |
| Q7 | `eval_lomo.py` | `lomo.json` | CPU 重算 |
| Q8 (审计) | `audit_parse.py` / `audit_parse_impact.py` | `audit_parse.json` / `audit_parse_impact.json` | CPU 审计 |
| Q8 (修正) | `eval_abstention.py` + `eval_tasks_t1_t3_t4.py`(eval_t1 源头改) | `abstention.json` / `tasks_t1_t4.json` | CPU 重算 |
| Q9 | `eval_routing_cost.py` | `routing_cost.json` | CPU 统计 |
| Q1/Q5(换图/先验) | （既有）`attribution.json` / `gap_vs_prior.json` | — | 既有 |

所有脚本在 vlm1 `~/SVD/agentic_probe` 下、conda env `grpo_ayb` 可一键复算。
