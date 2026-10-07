# GroundBind — ACL ARR Reviewer 14 条意见逐条回应

对应 reviewer 深度评审（gpt.md）列出的 14 条。按"已用数据闭环 / 需论文侧收紧措辞 / 待排期"三类标注。
所有新数字可由 `reviewer_v060/` 下脚本与 json 复算，分支 `exp/reviewer-rebuttal`。

---

## #1 Full framework 增益需在 matched retention 下比较 ⭐S级 — ✅ 已闭环

**reviewer 质疑**：Table 3 的 full vs support（FGR 22.55→19.16，−3.39pp；但 Rcorrect 94.26→91.71，−2.55pp）是 measured operating points，不是 matched retention。结构增益会不会只是"拒绝更多候选"？

**回应**：在**相同 achieved retention**（support@0.95 的实际保留率处，把 full 曲线插值到同一保留率）做配对比较，结构分支的真实增量是：

| 指标 | Δ(full − support) @ matched retention | 95% CI (模型级 bootstrap) | 显著 | 改善模型数 |
|---|---|---|---|---|
| **Relation FGR** | **−7.88pp** | [−9.43, −6.30] | ✅ SIG | **13/13** |
| **Relation CBR** | **−5.15pp** | [−6.73, −3.52] | ✅ SIG | 11/13 |
| ALL FGR | +0.16pp | [−0.56, +0.94] | ns | 6/13 |
| Positive mIoU | +0.001 | [−0.000, +0.002] | ns | 2/13 |

**结论**：
- 在**保留率完全匹配**（定位质量 mIoU 零损失，CI 跨零）下，relation FGR 仍降 **7.88pp 且 13/13 全改善**，显著 —— 增益**不是**来自更激进拒绝，而是结构分支对 relation 的真实判别。
- ALL FGR 匹配后 ≈0（ns）符合设计：object/co_occurrence/attribute 本就由 shared support 处理（见 #13），relation 才走结构分支。
- Table 3 看到的 3.39pp 是**未匹配 retention 的保守合并数字**；匹配后 relation 专项增益实为 7.88pp。

**产物**：`matched_retention_summary.json`、`matched_retention_current.json`、`eval_matched_current.py`。
**论文侧**：建议把 matched-retention relation ΔFGR/ΔCBR + CI 作为结构分支贡献的主证据（替代或补充 Table 3 的未匹配数字）。

---

## #5 语言×定位 query-level joint 统计 ⭐S级 — ✅ 已闭环（可能是最有冲击力的表）

**reviewer 质疑**：Figure 4 只报 marginal rates，没做 exact per-query cross-tab；joint grounding 本就同一响应内同时输出 support+box，为什么不直接测"说 unsupported 却仍画框"？

**回应**：按 (model, sample_id) 精确配对 t1(语言判断) × t2(定位行为)，在 24,000 条 negative query 上交叉制表：

| | grounding = empty | grounding = emits box |
|---|---|---|
| **says UNSUPPORTED** | 35.1% (correct refusal) | **26.7% ← LANGUAGE–GROUNDING CONTRADICTION** |
| **says SUPPORTED** | 2.0% (semantic FA only) | 30.6% (FA + grounding) |
| **ABSTAIN (parse-fail)** | 0.5% | 5.1% |

**核心数字：语言-定位矛盾率 = 26.72%** —— 模型明确判断"表达不被图像支持"，却仍为它输出一个候选框。

逐类矛盾率（都在 22–28%，非个别现象）：
- co_occurrence 28.47% · attribute 28.00% · object 27.80% · relation 22.60%

**结论**：这是论文标题 "Right Region, Wrong Reference" 和核心论点"能定位却 grounding 无效表达"的**最直接 per-query 证据**，强于 marginal rate 对比。建议升为正文主表。

**产物**：`joint_crosstab.json`、`eval_joint_crosstab.py`。
**论文侧**：新增一张 cross-tab 表（§5 诊断或 Figure 4 旁），正文报 26.72% contradiction rate。

---

## #6 Parser-fail 默认 NO 污染 FA/BA ⭐S级 — ✅ 已闭环（两版本已给）

**reviewer 质疑**：UniVG-R1 / TreeVGR 解析失败 37.24% / 24.84%，默认 NO 后在 negative 上看似"成功拒绝"，混淆语义拒绝与格式失败。必须补 strict（只认 explicit NO）与 current 两版本。

**回应**：两版本 FA/BA 对比（`abstention.json`）：

| 模型 | abstain% | current FA | strict FA | current BA | strict BA | ΔBA |
|---|---|---|---|---|---|---|
| **TreeVGR** | 24.84 | 45.80 | 66.23 | 76.60 | 66.78 | **−9.82** |
| **UniVG-R1** | 37.24 | 23.15 | 35.62 | 60.42 | 73.08 | **+12.66** |
| 其余 10 模型 | ≤0.84 | — | — | — | ≈ 不变 | ≈0 |

**结论**：
- 结论**在 10/12 模型上完全一致**（abstain%≤0.84，ΔBA≈0）；只有两个高解析失败模型需要修正，且两者方向相反（TreeVGR 虚高、UniVG-R1 虚低）——污染非常数非单向，无法靠统一声明抵消。
- 这两个模型的**高弃权率本身就是"语言判断不可靠"的直接证据**：定位通道正常输出框，语言通道大量解析失败——正是"语言-定位脱节"的极端案例。修正后论点不削弱反而更干净（不依赖被污染的 FA/BA）。

**产物**：`abstention.json`、`audit_parse_impact.json`、`eval_abstention.py`，源头修正已 commit `95a32c2`。
**论文侧**：Table 1 改用 strict/abstention 口径（FA/TPR/BA 在 parse_valid 子集算），abstain 率单列；Appendix A.1 从"披露"升级为"修正"。

---

## #4 外部泛化 + text-only / candidate-swap control ⭐S/A级 — ✅ text-only 已闭环

**reviewer 质疑**：若只给文本（不给图像）也能分 pos/neg，则 benchmark 有 lexical artifact。必须补 text-only control。

**回应**：喂固定中性灰图（无视觉信号）、保持 claim 文本与问句完全一致，让 OmniVerifier 只能靠文本：

- **AUROC(text-only) = 0.4486，95% CI [0.4382, 0.4595]** —— 上界 < 0.5
- 13/13 模型全在 0.44–0.46（均值 0.4475，范围 [0.4377, 0.4615]）
- 对照：有图像（nobox）AUROC 0.59–0.63

**结论**：去掉视觉信号后，纯文本**无法分离 pos/neg**（甚至略低于随机，因 negative 表达往往更长更具体、无图时模型倾向答 false）。**benchmark 不存在可被纯语言识别的 lexical artifact**，判别必须依赖图像。

**产物**：`textonly.json`、`collect_textonly.py`、`eval_textonly.py`（12,729 候选，0 错误，8-GPU 采集）。
**仍待（已排期外）**：candidate-swap control（已有 image-swap Δ11.41 可部分覆盖）、独立构造的 external set（FineCops-Ref / KnowDR-REC 兼容子集）——需新数据，列为后续。

---

## #9 CBR 跨模型均值 → common-success / own-elig 稳健性 A级 — ✅ 已闭环

**reviewer 质疑**：13-model mean CBR 分母是各模型自己的 positive-correct 子集，不是 common-difficulty leaderboard。

**回应**：
- **own-elig 大样本（每模型自己的 positive-correct 子集，n=131–275）**：**ROH > BOH 在 13/13 模型全部成立**；relation 最难在 10/13 模型。排序趋势**不是分母差异的伪影**。
- common-success 全交集（13 模型都 positive-correct）仅 7 图，样本太小不作主证据；趋势仍 ROH>BOH（10/13）。

**产物**：`cbr_common_bootstrap.json`、`eval_cbr_common_bootstrap.py`。
**论文侧**：报告"own-eligible 子集下 ROH>BOH 13/13"作为排序稳健性证据，说明 macro-mean 不是 leaderboard。

---

## #10 缺统计不确定性（结构增量仅 3.39pp 需 CI）A级 — ✅ 已闭环

**回应**：image-level paired bootstrap（source image 为重采样单元，2000 次）：
- #1 matched retention 全部 Δ 已带 95% CI（relation ΔFGR [−9.43,−6.30] 显著）。
- #9 每模型每类 CBR 带 95% CI（n≥131 的 11 模型 CI 宽度稳定 11–14pp；UniVG-R1/visual-rft 因 n≈30 CI 宽）。

**产物**：`matched_retention_summary.json`、`cbr_common_bootstrap.json`。
**论文侧**：主表 CBR / ΔFGR 补 image-group bootstrap 95% CI。

---

## #13 Structural verification scope 实际较窄 A级 — ✅ 已闭环（数据明确）

**reviewer 质疑**："Adaptive Structured Verification addresses four hallucination types" 放大了 scope，实际主要作用于 relation。

**回应**（`routing_cost.json`，32,500 候选）：
- 结构分支整体仅在 **20.97%** 候选上激活。
- 逐类激活率：**relation 44.75%** ≫ attribute 14.57% · co_occurrence 12.94% · object 9.66%。
- 增益也集中在 relation（matched retention 下 relation ΔFGR −7.88pp，ALL ΔFGR ≈0）。

**结论**：诚实表述为——**结构分支专门针对一组有语义对立的空间关系；object/co_occurrence/attribute 由 shared candidate support（一次 OmniVerifier 后验）处理**。这不是削弱，而是精确界定分工。
**论文侧**：§方法/§5 明确写 scoped 分工，不写"addresses four types"。

---

## #14 Computational cost 未交代 B级 — ✅ 调用预算已给（缺实测 latency）

**回应**（`routing_cost.json`）：
- support-only：每候选 2 次 scorer 调用（1 OmniVerifier + 1 JEV）。
- 结构分支 fired 时：额外 mean 2 次 OmniVerifier（rival predicates）→ 全局 **额外调用开销 41.94%**（13,632 extra calls / 32,500）。fallback_rate 11.11%。
- 因只有 20.97% 候选激活，平均开销远低于最坏情况 2+2K。

**论文侧**：补一张 call-budget 表（support 2 calls vs full 平均 2.84 calls）。实测 wall-clock latency / GPU-hours 仍待补（列 limitation 或后续）。

---

## 需论文侧收紧措辞（无需新实验）

### #3 方法是 per-model calibrated，非 universal zero-shot — ⚠️ 收紧 claim
附录事实：每个 upstream model × fold 单独 fit 一个 logistic head，用 positive/negative expression labels，image-level two-fold cross-fitting。
**建议**：正文统一写 **"model-specific calibrated post-hoc verification"**，不写 training-free generic verifier。理想补一个 transfer（held-out model 直接用共享 head），哪怕稍差也增强 generality——列为后续。

### #7 Gen vs RL 别作 Abstract headline — ⚠️ 改 framing
Abstract 现突出 general 37.64% vs RL 68.83% FGR，易被读成"RL 导致 grounding 幻觉"。
**建议**：headline 改"13 个 MLLM 的 false grounding 差异大、尽管 positive grounding 质量相近"；RL-high 作为 secondary observation 并明确 disclaim 非因果（backbone/训练任务/负监督/输出接口都不同）。

### #8 BOH/ROH 别写 "novel taxonomy" — ⚠️ 改定位
**建议**：写 **"diagnostic distinction tailored to grounding hallucination"**，真正 novelty 压到：same-image paired construction / positive-correct-conditioned CBR / aligned language+direct+joint grounding / mitigation at fixed candidate coordinates。Related Work 对 FineCops-Ref/CSR/KnowDR-REC 的区别需展开。

### #12 Table 2 "Ours RL 7B / selected for largest reduction" — ⚠️ 必改（cherry-pick 风险）
"从 13 个里挑 improvement 最大的" + "Ours RL 7B"（ASV 不是 RL 7B 模型）易被判 cherry-picking。
**建议（推荐）**：Table 2 只放 unfiltered 诊断，所有 mitigation 统一进 Table 3；或至少改标为 "UniVG-R1 + ASV（representative high-FGR case）"，删除 "largest reduction" 措辞。

---

## 待排期（成本高 / 非我方可独立完成）

### #2 Benchmark 标注质量审计 ⭐S级 — 用户线下处理
需人工独立复核统计：每类独立复核条数、double-annotation 比例、raw agreement、Cohen/Fleiss κ 或 Krippendorff α、disagreement/adjudication/repaired/excluded 数、per-type（Object/Association/Attribute/Relation）分列。**这是人工标注流程，由作者团队完成**；论文里把"remain required for release"改为给出已完成的统计。

### #11 External mitigation baseline B级
现只对比自身 support-only ablation。可补 cheap baseline：OmniVerifier-only / JEV-only / logistic 组合 / LLM-VLM binary judge / **matched-retention support-only（最低成本且已有，见 #1）**。其余需 GPU 排期。

### #4 后半 external dataset
FineCops-Ref / KnowDR-REC 兼容子集或新人工 100–200 图——需新数据采集，排期。

---

## reviewer 已认可、无需过度防御的点
- 已不把 Gen vs RL 写成 causal（正文+附录主动 disclaim）。
- CBR denominator 处理严谨（negative refusal 不从分母删，四类共享 positive-correct eligibility）。
- Positive grounding cost 未隐藏（同报 Rcorrect 与 all-positive mIoU）。
- predicate contrast 不当 logical proof（明确 competing predicates 可共存，margin 是 learned evidence）。

---

## 优先级总览（按 reviewer 建议顺序）

| # | 优先级 | 状态 | 关键结果 |
|---|---|---|---|
| 1 matched retention | S | ✅ | relation ΔFGR −7.88pp [−9.43,−6.30] 13/13 |
| 2 annotation audit | S | ⏳用户线下 | — |
| 5 joint cross-tab | S | ✅ | 语言-定位矛盾 26.72% |
| 4 text-only control | S/A | ✅ | AUROC 0.4486 [0.438,0.460] 无 artifact |
| 6 parser-fail 两版 | S | ✅ | 10/12 一致，仅 2 模型需修正 |
| 12 Table 2 Ours RL 7B | — | ⚠️措辞 | 建议删 Ours 行 |
| 10 bootstrap CI | A | ✅ | 全 Δ 带 image-level CI |
| 3 per-model calibrated | — | ⚠️措辞 | 收紧 claim |
| 9 CBR common subset | A | ✅ | ROH>BOH 13/13 own-elig |
| 13 structural scope | A | ✅ | 激活 20.97%，relation 44.75% |
| 8 external dataset | — | ⏳排期 | — |
| 14 latency / baseline | B | ◐ | 调用预算已给，缺 wall-clock |
| 7 Gen vs RL framing | — | ⚠️措辞 | 降为 secondary |
| 11 external baseline | B | ⏳排期 | matched support-only 已有 |
