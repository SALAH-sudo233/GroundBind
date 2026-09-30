# reviewer_v060 — v0.60 审稿意见补充实验

本目录回应 2026-09-30 的 NAACL 模拟审稿意见。**全部为 CPU 复算，未启动 GPU、未重跑任何模型推理**；
所需打分文件此前已存在于 vlm1 `~/SVD/agentic_probe/`，缺的是正确的汇总口径与缺失的对照分析。

## 先读

1. **`CORRECTION.md`** — 更正记录。先前报告的「另三类 FGR 显著变差」「ALL FGR 不显著」
   「方法只适用于有对立配置的空间关系」三项**已撤回**，根因是策略实现 bug（跨类型池化污染）。
2. `RESULTS_v060.md` — 主结果：P0-1 候选版本统一、P0-2 图文冲突定性、P0-3 同保留曲线与式(12) 可行性、
   P1-8 图像组级 CI、W1 路由收据。**逐类 FGR 与 ALL FGR 部分见 `CORRECTION.md` 的改正口径。**
3. `ATTRIBUTION_REPORT.md` — P1-6 归因三项对照。**结论部分已被 `CORRECTION.md` 覆盖**；
   三项对照本身有效，按论文定位降为附录级机制证据。

## 脚本（运行于 vlm1 `~/SVD/agentic_probe/`）

| 脚本 | 作用 | 输出 |
|---|---|---|
| `audit_routing_fixed.py` | 修复原审计的 records 读取 bug（原脚本计数器全为 0，路由合法性此前无收据） | `audit_routing_fixed.json` |
| `eval_unified.py` | **逐模型**解析候选来源，产出一致的 13 模型纠正候选主表 | `unified_13models_095.json` |
| `eval_matched.py` | 保留率扫描 + 同保留匹配比较 + 式(12) 可行带 + 图像聚类 bootstrap | `matched_retention.json` |
| `eval_attribution.py` | [A] 词先验 [B] 等预算改写 [C] 错图对照 | `attribution.json` |
| `diag_gap_vs_prior.py` | gap 与谓词身份的冗余性诊断（ANOVA R²、残差 AUROC、词内 AUROC） | `gap_vs_prior.json` |
| `eval_scoped.py` | 按「谓词是否有真实对立配置」分作用域重做决定性比较 | `scoped.json` |
| `diag_type_gate.py` | **四策略对照 + 路由逐类触发率 + 逐类 gap AUROC**（定位池化污染） | `type_gate.json` |
| `diag_final_policy.py` | **support / A_all / A_reject / A_reject_opp 最终口径** | `final_policy.json` |
| `export_cf_cbr.py` | 两个纠正坐标模型的**四类未过滤 CBR**（补论文横线列） | `cf_cbr_4types.json` |
| `eval_paper_tables.py` | **论文主表复算**：11 模型面板 + 纠正坐标模型附录，带未过滤 CBR 校验门 | `paper_tables.json` |

## 论文主表复算（target 0.95）

`paper_tables.json` 是主表的仓库内凭据 —— 此前论文 Table 3 的数字在服务器上**查不到出处**
（`grep` 整个 `~/SVD` 只命中 trainer_state 与坐标数据里的巧合子串）。

**候选一致面板（11 模型，Table 3）**

| 配置 | FGR | obj CBR | cooc CBR | attr CBR | rel CBR | 正确保留 | mIoU |
|---|---|---|---|---|---|---|---|
| 未过滤 | 58.36% | 15.61% | 25.59% | 33.84% | 42.11% | 100.00% | 0.4221 |
| 仅支持核验 | 22.98% | 1.76% | 9.23% | 8.57% | 26.83% | 94.33% | 0.3971 |
| 自适应结构化核验 | **19.30%** | 1.76% | 9.16% | 8.50% | **18.56%** | 91.51% | 0.3860 |

**纠正坐标模型（2 模型，附录 C）**

| 配置 | FGR | obj CBR | cooc CBR | attr CBR | rel CBR | 正确保留 | mIoU |
|---|---|---|---|---|---|---|---|
| 未过滤 | 64.03% | 21.19% | 25.85% | 34.64% | 46.52% | 100.00% | 0.4525 |
| 仅支持核验 | 20.15% | 1.26% | 9.19% | 6.94% | 28.83% | 93.93% | 0.4231 |
| 自适应结构化核验 | 18.40% | 1.26% | 8.56% | 6.50% | 23.62% | 92.82% | 0.4184 |

**与 PDF 的差异**：前两行逐位复现（最大偏差 0.018pp，舍入）；**第三行全部不一致且每项更好**
（FGR 21.55 → 19.30、attr CBR 10.21 → 8.50、rel CBR 19.29 → 18.56、保留 91.99 → 91.51）。
PDF 第三行让探针头决定所有被路由的行 → 跨类型池化污染，正是 attr CBR 从 8.55 反弹到 10.21 的来源。
本复算用单调门 + 作用域门，被门挡住的行逐位沿用支持核验决策。

**校验门**：脚本强制未过滤 CBR 与独立导出的 `cf_cbr_4types.json` 一致
（38.6/37.0/50.8/55.1 与 3.8/14.7/18.5/37.9），不通过则拒绝采信。已 PASS。

## 纠正坐标后的四类未过滤 CBR（%）

论文 Table 2 / Figure 3 / Table A5 只给了这两个模型的 relation CBR，另三类是横线。补齐如下
（relation 逐位复现已发表的 55.1 / 37.9，作为导出正确性的校验锚点）：

| 模型 | n_c | object | co_occurrence | attribute | relation |
|---|---|---|---|---|---|
| UniVG-R1 | 254 | **38.6** | **37.0** | **50.8** | 55.1 ✓ |
| Visual-RFT | 211 | **3.8** | **14.7** | **18.5** | 37.9 ✓ |

口径原样继承 `eval_cbr_paper_aligned.py`，未加任何条件。
Visual-RFT 呈现完整难度阶梯（3.8 → 14.7 → 18.5 → 37.9），与 §4.2 论调一致；
UniVG-R1 四类偏平（38.6 / 37.0 / 50.8 / 55.1），因其对 99.8% 负例出框、几乎不拒绝。
旧坐标口径的同一计算（UniVG-R1 33.3/46.7/40.0/46.7 n_c=30；Visual-RFT 0.0/11.5/19.2/11.5 n_c=26）
样本量仅 26–30、难度阶梯不成立，**只作版本审计，不进主表**。

依赖：`eval_upstream.py`、`eval_cbr_paper_aligned.py`、`simple_relations.py`、`canon_roots_paper.json`、
`canon_roots_coordfix.json` 及既有打分文件（见 `RESULTS_v060.md` 末尾清单）。

## 四条最重要的结论

1. **主表可以统一到纠正后候选。** UniVG-R1 n_c 30→254、visual-rft 26→211，13 模型一张表。
   机制根因：`--coordfix` 是全局布尔，而纠正打分文件只存在于两个模型 → 只能产出「13模型旧坐标」
   或「2模型新坐标」。修法是**逐模型解析候选来源**。

2. **推荐口径 `A_reject_opp` 下四类幻觉全部显著改善**（单调门 + 作用域门，target 0.95，13 模型）：
   object −0.108pp、co_occurrence −0.415pp、attribute −0.200pp、relation −12.831pp，
   **ALL FGR −3.388pp CI[−4.035,−2.754] 13/13 显著**，relation CBR −7.795pp 13/13，
   实际保留 0.9171。**没有任何一类变差**，与 v0.62「缓释效果覆盖四类而非仅限关系」一致。

3. **同一校准 target ≠ 同一工作点。** 同 target 0.95 下 full 臂实际保留 0.9219 vs support 0.9426。
   在 support 的实际保留率上内插 full 曲线：关系 FGR −5.87pp CI[−8.46,−3.25]、
   关系 CBR −4.68pp CI[−6.41,−2.90] (11/11)。约一半原始差值来自工作点更保守。
   Orsta-7B/TreeVGR 匹配点落在观测区间外，**不外推**，故报 11/13。

4. **路由合法但非 relation-only。** 原 `audit_routing.py` 计数器全零（读错 records 路径），
   「路由合法」此前零收据。修复后：文本路由确实只读 query + 冻结词表（合法），但在四类上都触发
   （object 0.238 / cooc 0.326 / attr 0.349 / relation 0.490）→ 所以必须用**单调门**保护另三类，
   而不是靠非法元数据门。旧报告「另三类恰好 0.00pp 不变」是非法标签门的假象。

## 不可引用的旧结论

- 「object/co_occurrence/attribute FGR 显著变差」「ALL FGR 不显著」「同保留下 ALL FGR +2.05pp」
  —— **策略 bug 产物**，见 `CORRECTION.md`。前三类只走一次 Omni-7B 后验，不经结构化词表竞争打分，
  完整臂对它们应是严格增量。
- 「方法只适用于存在语义互斥对立配置的空间关系」 —— **与论文论调相反**，已撤回。
  作用域门是为保护另三类不被池化污染，不是限制适用范围。
- 「缓释只在 relation 触发、另三类恰好 0.00pp 不变」 —— 非法标签门的产物。
- `AUDIT.md` 第 1 条应更新：`keep_a2()` 不要求 `keep_base` 先通过，代码与式(10) 一致，**冲突在 Figure 4**。

## 单谓词数值的引用限制

`on top of` n_pos=11、`under` n_pos=39、`above` n_pos=26，单谓词 AUROC 不可当独立良好估计引用；
`beside`(72/0)、`below`(19/0)、`on`(13/0) 零正例，不贡献判别。
承载权重的是 `behind`(n_pos=270) 与 `in front of`(n_pos=1145)。
