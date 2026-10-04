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
| `eval_matched.py` | 保留率扫描 + 同保留匹配比较 + 式(12) 可行带 + 图像聚类 bootstrap | `matched_retention_WITHDRAWN_A_all.json`（旧策略，已撤回） |
| `eval_matched_current.py` | **当前策略** `A_reject_opp` 的同保留对照（单调门+作用域门） | `matched_retention_current.json` / `E3_MATCHED_RETENTION.md` |
| `eval_attribution.py` | [A] 词先验 [B] 等预算改写 [C] 错图对照 | `attribution.json` |
| `diag_gap_vs_prior.py` | gap 与谓词身份的冗余性诊断（ANOVA R²、残差 AUROC、词内 AUROC） | `gap_vs_prior.json` |
| `eval_scoped.py` | 按「谓词是否有真实对立配置」分作用域重做决定性比较 | `scoped.json` |
| `diag_type_gate.py` | **四策略对照 + 路由逐类触发率 + 逐类 gap AUROC**（定位池化污染） | `type_gate.json` |
| `diag_final_policy.py` | **support / A_all / A_reject / A_reject_opp 最终口径** | `final_policy.json` |
| `export_cf_cbr.py` | 两个纠正坐标模型的**四类未过滤 CBR**（补论文横线列） | `cf_cbr_4types.json` |
| `eval_paper_tables.py` | **论文主表复算**：13 模型主面板 + 11 模型对账 + 纠正坐标附录，带未过滤 CBR 校验门 | `paper_tables.json` |
| `render_paper_tables.py` | **按论文指标名渲染 Table 2 / Table 3**（BOH/ROH、正例成功率、家族聚合），自带四道校验门 | 终端输出 |
| `eval_tasks_t1_t3_t4.py` | **另三项任务评测**：表达核验 / 图像描述 / 联合定位，带 §4.1 硬门 | `tasks_t1_t3_t4.json` |
| `export_cf_joint.py` | **纠正坐标后的联合定位**（M2 的 12 个缺项），带 t2 复现硬门 | `cf_joint_grounding.json` |
| `collect_t4_mitigation.py` | **t4 验证器打分采集**（Omni 原判 + 竞争探针 + JEV 头），8 卡分片 | `t4mitig/t4_shard*.jsonl` |
| `eval_t4_mitigation.py` | **t4 缓释评测**，带四类严格增量检查 | `t4_mitigation.json` |

## t4 联合定位缓释（详见 `T4_MITIGATION.md`）

缓释框架此前只在 t2 上跑过。t4 现已补齐 13 模型完整结果。

**t2 打分不可复用**：`collect_jevhead.py` / `collect_probe_textroute.py` 都硬过滤
`task == 't2_vqa_grounding'`；验证器打的是具体框的分，而 t4 框与 t2 框不同
（多产出描述会让框移动）。复用等于跨对象混用分数。重新采集 **59,025 次前向**
（8 卡 25.4 分钟，零错误，计数逐项吻合计划）。

| 配置 | FGR | BOH | ROH | 正确保留 | mIoU |
|---|---|---|---|---|---|
| 未过滤 | 48.24% | 36.46% | 60.02% | 100.00% | 0.3639 |
| 仅支持核验 | 19.70% | 10.52% | 28.87% | 93.95% | 0.3379 |
| **自适应结构化核验** | **18.45%** | **10.05%** | **26.85%** | **92.99%** | 0.3341 |

**四类零回退、13/13 模型全改善**：object −0.231pp / co_occurrence −0.723pp /
attribute −1.046pp / relation −2.985pp，四个 CI 全不含 0，变差模型数均为 **0/13**。
这是「前三类只走一次 Omni 后验、必须严格增量」这一要求在 t4 上的验证。

**缓释在 t4 上代价更低**：完整臂 FGR 18.45% < t2 的 19.16%，正确保留 92.99% > t2 的 91.71%
→ 框架不是只对直接定位有效。

## M2 补全：纠正坐标后的联合定位

详见 `JOINT_GROUNDING_CF.md`。要点：`write_rescaled_run.py` 的过滤器是
`T2 = ('t2','t2_vqa_grounding')`，**坐标纠正从未作用于 t4** —— 实测 coordfix run 里两个
模型的 t4 框与未纠正 run 逐位相同（2500/2500），而 t2 分别差 2496 与 974 行。

| 模型 | n_correct | mIoU | CBR obj | CBR cooc | CBR attr | CBR rel |
|---|---|---|---|---|---|---|
| UniVG-R1 | 247 | 0.4820 | 25.91% | 34.41% | 47.37% | 55.06% |
| Visual-RFT | 207 | 0.4249 | 17.39% | 24.64% | 35.27% | 49.76% |

校验门：同一代码路径限制到 t2 必须复现已发表的 38.6/37.0/50.8/55.1 与
3.8/14.7/18.5/37.9，两模型全部命中 PASS，否则 `exit 1` 拒绝输出 t4。

**Visual-RFT 的联合定位框复用全面恶化**（object 3.79 → 17.39，4.6 倍），
是「同时生成语言与区域会改变接受行为」在 CBR 上的直接证据。

## 四项任务凭据（此前只有 t2）

论文 Table 1 定义四项任务，但仓库长期只能为**直接定位（t2）**背书。另三项的数据一直在
服务器上（t1/t4 各 13 模型 × 2,500 行，t3 各 500 行），**从未有脚本读过**。现已补全：

```sh
python3 eval_tasks_t1_t3_t4.py --json-out tasks_t1_t3_t4.json   # vlm1，纯 CPU，约 2 分钟
```

**硬门**：t1 两个家族的错误接受率必须复现论文 §4.1 印出的 29.44% / 33.03%。
实测 29.4375% / 33.0278%，双门 PASS。

**跨 run 解析**：`canon_roots_paper.json` 指向的是各模型 t2 规范框所在 run，但该 run
未必带其他任务 —— **Qwen3-VL-8B 的 t1 不在其中**（在 `refcocog_eval_11models_500_repaired`），
而 t3 只存在于 `refcocog_eval_13models_4tasks_500`。脚本逐任务独立解析并记录 `run_provenance`。
这一点直接影响结论：漏掉 Qwen3-VL-8B 会让通用家族错误接受率变成 31.35%（12 模型），
与论文的 29.44% 对不上。回退时**拒绝 500 行以下的候选**，避免误取 40 行的 smoke run。

| 任务 | 关键结果（全部 13 模型） |
|---|---|
| t1 表达核验 | 错误接受 31.92%、平衡准确率 76.98%、ROH 44.12% ≫ BOH 19.72% |
| t3 图像描述 | 对象幻觉 4.29%、AMBER 余弦 0.4801 |
| t4 联合定位 | FGR 48.25%、正例成功 35.60%、mIoU 0.3639、描述幻觉 10.07%（两模型已纠正坐标） |

两处值得写进论文的观察：

- **语言与空间的差距**：错误接受率家族差仅 3.59pp，错误出框率家族差 **31.19pp**。
- **LENS 在 t4 的描述幻觉 55.60%** 是明显异常（其他模型 0.45–14.55%），而其 t3 无查询
  描述幻觉仅 4.20%。同一模型在联合任务下描述崩塌，引用 t4 描述指标须单独说明。
- **UniVG-R1 的 t1 解析失败率 37.24%**、TreeVGR 24.84%，其余多为 0。这两个模型的
  平衡准确率（60.42% / 76.60%）受解析影响，不宜与 0 失败率的模型直接并列。

## 按论文指标查看两张表

`paper_tables.json` 存的是逐模型分臂结果，而论文印的是 BOH / ROH / 正例成功率 / 家族聚合，
且 `pooled` 块**根本不含 BOH/ROH**。这两张表此前靠手工拼装 —— 别人复不出来。改用脚本：

```sh
python3 render_paper_tables.py                  # 13 模型主面板
python3 render_paper_tables.py --panel table3    # 11 模型，与 PDF 对账
python3 render_paper_tables.py --panel appendixC # 纠正坐标两模型
```

纯本地运行，不需要服务器和 GPU。它**重算而非信任**：BOH/ROH 从 `by_ht` 重新推导、
pooled 从逐模型重新平均、家族聚合对齐论文 §4.1 印出的数字，任一不符即 `exit 1` 并拒绝出表。

四道门实测（篡改注入验证，确认门会失败而非空转）：

| 注入 | 结果 |
|---|---|
| 未篡改 | exit 0，all gates PASS |
| 单模型 `boh` +0.01 | exit 1，`BOH/ROH do not match by_ht` |
| `pooled.full.FGR` +0.5 | exit 1，`pooled block disagrees with per-model mean` |
| 单模型 `pos_miou` +0.01 | exit 1，同上（经家族聚合放大后被捕获） |
| 上游 `unfiltered_cbr_gate_passed=False` | exit 1，拒绝读入 |

§4.1 家族统计逐位复现论文原文：通用 mIoU 0.4331 / 正例成功 45.0% / FGR 37.64%，
RL 适配 0.4240 / 42.9% / 68.83% —— 这佐证仓库这批 13 模型数据就是论文所用那批。

## 论文主表复算（target 0.95）

`paper_tables.json` 是主表的仓库内凭据 —— 此前论文 Table 3 的数字在服务器上**查不到出处**
（`grep` 整个 `~/SVD` 只命中 trainer_state 与坐标数据里的巧合子串）。
文件含三个面板：`main13`（推荐主表）、`table3`（PDF 对账）、`appendixC`（纠正坐标模型）。

**统一 13 模型面板（`main13`，推荐）** —— 评测与缓释同口径，合格正例合计 2,832

| 配置 | FGR | obj CBR | cooc CBR | attr CBR | rel CBR | 正确保留 | mIoU |
|---|---|---|---|---|---|---|---|
| 未过滤 | 59.23% | 16.47% | 25.63% | 33.96% | 42.79% | 100.00% | 0.4268 |
| 仅支持核验 | 22.55% | 1.69% | 9.22% | 8.32% | 27.14% | 94.26% | 0.4011 |
| 自适应结构化核验 | **19.16%** | 1.69% | 9.07% | 8.19% | **19.34%** | 91.71% | 0.3910 |

逐类 FGR：object 39.37 → 6.18 → 6.08、co_occurrence 56.85 → 18.22 → 17.80、
attribute 63.60 → 17.29 → 17.09、relation 77.12 → 48.49 → 35.66（%）。

与独立脚本 `diag_final_policy.py` 的 `A_reject_opp` 臂**逐位一致**（14 项指标 delta 全为
0.0000），两条不同代码路径的交叉验证。

**候选一致面板（11 模型，Table 3，仅用于与 PDF 对账）**

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
