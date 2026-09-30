# vlm1_archive — vlm1 服务器实验产物归档

来源：`vlm1:~/SVD/agentic_probe`（原目录 208 个文件 / 96MB）。
本目录只收**代码、报告、结果摘要**（71 文件 / 1.0MB）；92MB 的逐样本打分 `.jsonl` 和 51 个运行日志留在服务器。
策划依据与排除理由见 `MANIFEST.txt`。凭据扫描零命中。

## 目录

| 路径 | 内容 |
|---|---|
| `code/` | 39 个分析/采集脚本与运行封装（已去掉与 `method/`、`reviewer_v060/` 逐位相同的 18 个） |
| `reports/` | 3 份权威报告 |
| `reports/historical/` | 5 份早期阶段报告，部分结论已被取代 |
| `results/` | 10 个权威结果摘要 + `tables/` 文本渲染 |
| `results/version_audit/` | 10 个被取代或有缺陷的聚合结果，**仅作溯源，不得用于主表** |

## 报告权威性

| 文件 | 日期 | 地位 |
|---|---|---|
| `reports/REVIEWER_EVIDENCE.md` | 09-30 | **最新**。W1/W2/W3 审稿论据；W2 推翻了论文当时的一个头条发现（"专用模型 94% 正例定位失败"实为坐标口径 bug） |
| `reports/CBR_PAPER_ALIGNED_RESULTS.md` | 09-27 | CBR 口径对齐基准，13/13 模型逐位复现 v0.48 Table 4 |
| `reports/AGENTIC_H2H3_RESULTS.md` | 09-27 | H2 否证、H3 基本否证、H1 再次成立 |
| `reports/historical/*` | 09-27 | 阶段记录。各文件头部自述了本身的作废数字，引用前先读那段 |

`MITIGATION_FRAMEWORK_REPORT.md` 未收在此处 —— 它与 `evidence/` 下已跟踪的版本逐位相同。

## 两个刻意保留的"错误证物"

1. **`code/audit_routing_BROKEN_readpath.py`** — 原审计脚本。它读
   `<root>/<model>/<task>/records.jsonl`，而真实布局是 `<root>/<model>/records.jsonl` 单文件带 `task` 字段，
   于是所有计数器停在 0，`audit_routing.json` 全是空 dict，"路由合法"这一断言此前**零收据**。
   修复版在 `reviewer_v060/audit_routing_fixed.py`。
   **教训：审计脚本输出全零/空容器时，先验证它的读取路径，不要记为 PASS。**

2. **`results/version_audit/`** — 含 `main_fixedbudget_095.json`（13 模型但 `coordfix=False`）与
   `cf_fixedbudget_095.json`（仅 2 模型纠正坐标）。这两份**割裂表**正是审稿意见 P0-1 的物证：
   `eval_combo_fixedbudget.py` 的 `--coordfix` 是全局布尔，而纠正打分文件只存在于 UniVG-R1 / visual-rft，
   所以它永远产不出一致的 13 模型表。修法见 `reviewer_v060/eval_unified.py`（逐模型解析候选来源）。

## 服务器上刻意未收的两类文件

- **`probecf_omni_WRONGSCOPE.jsonl`** — 作用域错误：按全部 13 模型采集，而坐标纠正只适用两个模型，
  白跑 4600 行。正确文件是 `probecf_omni.jsonl`。`REVIEWER_EVIDENCE.md` 有过程记录。
- **`r1probe_shard*.jsonl`** — **不可当冗余删**。`r1probe_all.jsonl` 6034 行，而分片合计 13562 行，
  说明 all 是**过滤后的子集**而非简单拼接。其余 6 组分片经逐行核对与合并文件一致，故未收。

## 复现入口

脚本默认从 `~/SVD/agentic_probe` 读打分文件，直接在服务器该目录下运行。
纯 CPU 复算（不需要 GPU）：

```sh
python3 audit_routing_fixed.py      # 路由合法性收据
python3 eval_unified.py --target 0.95 --json-out unified_13models_095.json
python3 eval_matched.py --base-target 0.95 --json-out matched_retention.json
python3 diag_type_gate.py           # 四策略对照 + 逐类路由触发率
python3 diag_final_policy.py        # support / A_all / A_reject / A_reject_opp
python3 export_cf_cbr.py            # 两个纠正坐标模型的四类 CBR
```

需要 GPU 的采集脚本（`collect_*.py` + 对应 `run_*.sh`）已收录但**无需重跑** —— 其产物就是留在服务器的 `.jsonl`。

## CBR 契约（复算前必读）

θ=0.5 判正例正确、ρ=0.8 相似度**对正例预测框**计算、分母为共享合格数 `n_c`、
拒绝计零但留在分母、`c_i=0` 整组移出。`eval_cbr_paper_aligned.py` 有 assert 强制四类共享同一合格集。
**重算已发表指标时不要自行添加条件**（例如排除退化框），否则与论文对不上。
