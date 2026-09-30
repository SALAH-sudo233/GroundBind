# evidence/ — v0.51 继承结果（部分已作废）

本目录保存 v0.51 阶段的结果文件，**逐位保留未修改**，用于溯源。
其中一部分数字**不可引用**：它们出自后来被证实非法的门控。当前权威数字在
`reviewer_v060/paper_tables.json`，主表见仓库根 `README.md`。

## 不可引用的数字（出自本目录）

`MITIGATION_FRAMEWORK_REPORT.md`（2026-09-27）与 `combo.json` / `combo.txt` 报告的
**ALL FGR 59.20% → 22.15%、relation CBR 40.1% → 18.7%、positive mIoU 0.3738 → 0.3549**
是在**非法元数据门控**下得到的 —— 过滤器读了 `hallucination_type` 真值标签来决定只在
relation 行改判。这既不是部署时可得的信息，也造成了「另三类恰好 0.00pp 不变」的假象。

合法的查询文本路由在**四类上都会触发**（object 0.238 / co_occurrence 0.326 /
attribute 0.349 / relation 0.490），收据见 `reviewer_v060/audit_routing_fixed.json`。

同一目录的 `cbr_3lines.json`、`cbr_fairness.json`、`final_report.json` 属于当时的
三条缓释线与公平性对照，口径与当前主表不同，引用前先核对分母与门控定义。

## 当前权威对应关系

| 想要的数字 | 去哪里取 |
|---|---|
| 主表三行（11 模型面板） | `reviewer_v060/paper_tables.json` → `table3.pooled` |
| 纠正坐标模型缓释（附录 C） | `reviewer_v060/paper_tables.json` → `appendixC.pooled` |
| 两个纠正坐标模型的四类未过滤 CBR | `reviewer_v060/cf_cbr_4types.json` |
| 13 模型统一候选表 | `reviewer_v060/unified_13models_095.json` |
| 同保留率曲线与匹配比较 | `reviewer_v060/matched_retention.json` |
| 路由合法性收据 | `reviewer_v060/audit_routing_fixed.json` |
| 策略对照（证明单调门保护另三类） | `reviewer_v060/final_policy.json`、`type_gate.json` |
| 撤回记录与根因 | `reviewer_v060/CORRECTION.md` |

## 为什么不删

这些文件是审计证据链的一部分：`AUDIT.md` 与 `CORRECTION.md` 的结论都引用它们作为
「当时报告了什么」的对照。删掉会让更正记录失去被更正的对象。
