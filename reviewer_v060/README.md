# reviewer_v060 — v0.60 审稿意见补充实验

本目录回应 2026-09-30 的 NAACL 模拟审稿意见。**全部为 CPU 复算，未启动 GPU、未重跑任何模型推理**；
所需打分文件此前已存在于 vlm1 `~/SVD/agentic_probe/`，缺的是正确的汇总口径与缺失的对照分析。

## 先读

- `RESULTS_v060.md` — 主结果：P0-1 候选版本统一、P0-2 图文冲突定性、P0-3 同保留曲线与式(12) 可行性、P1-8 图像组级 CI、W1 路由收据
- `ATTRIBUTION_REPORT.md` — P1-6 归因三项对照，含**两个否证**与方法适用范围的收缩

## 脚本（运行于 vlm1 `~/SVD/agentic_probe/`）

| 脚本 | 作用 | 输出 |
|---|---|---|
| `audit_routing_fixed.py` | 修复原审计的 records 读取 bug（原脚本计数器全为 0，路由合法性此前无收据） | `audit_routing_fixed.json` |
| `eval_unified.py` | **逐模型**解析候选来源，产出一致的 13 模型纠正候选主表 | `unified_13models_095.json` |
| `eval_matched.py` | 保留率扫描 + 同保留匹配比较 + 式(12) 可行带 + 图像聚类 bootstrap | `matched_retention.json` |
| `eval_attribution.py` | [A] 词先验 [B] 等预算改写 [C] 错图对照 | `attribution.json` |
| `diag_gap_vs_prior.py` | gap 与谓词身份的冗余性诊断（ANOVA R²、残差 AUROC、词内 AUROC） | `gap_vs_prior.json` |
| `eval_scoped.py` | 按"谓词是否有真实对立配置"分作用域重做决定性比较 | `scoped.json` |

依赖：`eval_upstream.py`、`eval_cbr_paper_aligned.py`、`simple_relations.py`、`canon_roots_paper.json`、
`canon_roots_coordfix.json` 及既有打分文件（见 `RESULTS_v060.md` 末尾清单）。

## 三条最重要的结论

1. **主表可以统一。** UniVG-R1 n_c 30→254、visual-rft 26→211，13 模型一张表；关系 FGR −11.03pp、
   关系 CBR −6.60pp 均 13/13 显著，但 object/co_occurrence/attribute 三类 FGR **全部显著变差**，ALL FGR 不显著。
2. **同保留下增量仍在，但减半。** 关系 FGR −5.87pp CI[−8.46,−3.25]、关系 CBR −4.68pp CI[−6.41,−2.90] (11/11)；
   同保留下 ALL FGR 反而 **+2.05pp 显著上升**。约一半原始差值来自工作点更保守。
3. **方法范围必须收缩。** 叠加词先验后池化增量为零（−0.0015）；分作用域后，有真实对立配置的谓词族
   +0.0059 CI[+0.0026,+0.0090] 显著 11/13，而 `next to` 幅度可忽略。原因是 `next to` 无逻辑对立词
   （谓词身份只解释 gap 方差的 8%，故非冗余），词内 gap AUROC 0.5496 vs behind 0.8069 / under 0.8898。

## 不可引用的旧结论

- 旧报告"缓释只在 relation 触发、另三类恰好 0.00pp 不变"是**非法标签门的产物**：合法文本路由在四类上都会触发。
- `AUDIT.md` 第 1 条应更新：`keep_a2()` 不要求 `keep_base` 先通过，代码与式(10) 一致，**冲突在 Figure 4**。
