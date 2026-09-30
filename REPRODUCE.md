# 全流程复现清单

**全流程要素都在 vlm1 上，不只是结果摘要。** 本文件记录 2026-10-01 的实测核验：
从图像与上游候选出发，重跑采集 → 汇总 → 出表的每一环，所需文件都存在且路径已验证。

早先 `AUDIT.md` 第 4 条写「本 checkout 缺少 records.jsonl、打分缓存、权重、图像」——
那句话对**仓库**成立，但会被误读成「全流程不可复现」。实际上服务器侧完整。
本文件纠正这一印象并给出确切路径。

主机 `vlm1`（别名见 `~/.ssh/config`），工作目录 `~/SVD/agentic_probe`。
实测时 8×RTX4090 全部空闲（0 MiB / 24564 MiB）。

## 一、要素核验（2026-10-01 实测）

| 要素 | 路径 | 状态 |
|---|---|---|
| 基准规格（500 组 / 2500 项） | `~/benchmark/repaired/refcocog_500_dev.semantic_strict.json` | ✅ |
| 图像 | `~/models/LENS/data/refcoco/train2014` | ✅ 82,783 张 |
| 上游候选 `records.jsonl` | 见 `canon_roots_paper.json` 逐模型映射 | ✅ **13/13 全部存在** |
| 纠正坐标候选 | `canon_roots_coordfix.json`，UniVG-R1 / visual-rft 指向 `~/benchmark/coordfix_500` | ✅ 13/13 |
| JEV-2B 基座 | `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B` | ✅ |
| JEV-2B LoRA adapter | `~/SVD/grpo_verifier/runs/jev_verifier_v2/adapter_final/` | ✅ 10,045,000 B |
| JEV-2B 标量读数头 | 同目录 `head_final.pt` | ✅ 10,173 B |
| JEV-2B 温度 | 同目录 `temperature.json` | ✅ **T = 1.7765671239028529** |
| OmniVerifier-7B（冻结） | `~/.cache/huggingface/hub/models--comin--OmniVerifier-7B` | ✅ |
| 13 个上游模型权重 | `~/.cache/huggingface/hub/models--*` | ✅ |

温度标定元数据（`temperature.json` 原样）：`split=calibration`、`n=512`、
`nll 0.15250 → 0.11968`、`ece 0.03053 → 0.01137`。

**注意路径易错点**：采集脚本里是 `~/SVD/grpo_verifier/runs/jev_verifier_v2`，
中间有 `runs/`。少写这一层会得到「文件不存在」的错误结论 —— 我核验时就先踩了一次。

## 二、已有打分文件（重跑采集前先确认是否真需要）

| 文件 | 行数 | 体积 | 用途 |
|---|---|---|---|
| `probe_textroute_all.jsonl` | 25,432 | 7.9M | Omni 支持分 + 文本路由探针 |
| `trprobe_jev_all.jsonl` | 25,432 | 7.4M | JEV 头对应行 |
| `probecf_omni.jsonl` | 4,096 | 1.4M | 纠正坐标 Omni（仅两模型） |
| `probecf_jev.jsonl` | 4,096 | 1.3M | 纠正坐标 JEV |
| `actions_relation.jsonl` | 2,209 | 764K | 等预算改写对照 |
| `swap_all.jsonl` | 13,562 | 4.2M | 错图对照（`z_real`/`z_swap`） |

这些是 GPU 采集的产物。**主表复算不需要重跑它们**，纯 CPU 即可。

## 三、复现层级

### L1 出表（本地，秒级，无需服务器）

```sh
cd reviewer_v060
python3 render_paper_tables.py                   # Table 2 / Table 3，论文指标名
python3 render_paper_tables.py --panel table3     # 11 模型，与 PDF 对账
python3 render_paper_tables.py --panel appendixC  # 纠正坐标两模型
```

自带四道门：BOH/ROH 从 `by_ht` 重推、pooled 从逐模型重算、家族聚合对齐论文 §4.1、
上游校验门标记。任一不符 `exit 1`。已做篡改注入验证（五种注入全部被拦）。

### L2 汇总复算（vlm1，纯 CPU，分钟级）

```sh
cd ~/SVD/agentic_probe
python3 eval_paper_tables.py --target 0.95 --json-out paper_tables.json   # ~4 min
python3 export_cf_cbr.py                 # 两模型四类未过滤 CBR
python3 diag_final_policy.py             # 四策略对照
python3 eval_matched.py --base-target 0.95 --json-out matched_retention.json
python3 audit_routing_fixed.py           # 路由合法性收据
```

读上表的打分文件，不加载模型、不占 GPU。`eval_paper_tables.py` 内建校验门：
未过滤 CBR 必须匹配独立导出的 `cf_cbr_4types.json`，否则明确标记 FAIL。

### L3 采集重跑（vlm1，需 GPU）

```sh
cd ~/SVD/agentic_probe
bash run_textroute_omni.sh   # OmniVerifier-7B 支持分 + 文本路由
bash run_textroute_jev.sh    # JEV-2B 头
bash run_coordfix_probe.sh   # 纠正坐标两模型
bash run_swap.sh             # 错图对照
bash run_jevhead.sh          # JEV 头独立采集
```

采集脚本：`collect_probe_textroute.py`、`collect_jevhead.py`、`collect_aswap.py`、
`collect_actions.py`、`collect_r1_probe.py`、`collect_2b_select.py`。
上游候选生成用 `deploy_upstream.py` + `run_deploy.sh`。

**L3 会覆盖 L2 的输入文件。** 重跑前先备份现有 `.jsonl`，否则无法与已发表数字对账。

### L4 JEV-2B 训练

训练源在仓库 `training/`，产物即上表的 adapter/head/temperature。
**复现主表不需要这一层** —— 现成 checkpoint 已在服务器上。

## 四、验收锚点（改动后必须仍然成立）

任何重算都要先复现这些已发表值，否则说明口径被改动了：

| 锚点 | 值 | 来源 |
|---|---|---|
| UniVG-R1 relation CBR（纠正坐标，未过滤） | **55.1%** | 论文 Table 2 / `export_cf_cbr.py` |
| visual-rft relation CBR（同上） | **37.9%** | 同上 |
| 通用家族 mIoU / 正例成功 / FGR | **0.4331 / 45.0% / 37.64%** | 论文 §4.1 |
| RL 适配家族同三项 | **0.4240 / 42.9% / 68.83%** | 论文 §4.1 |
| 未过滤 FGR（13 模型） | **59.23%** | `paper_tables.json` → `main13` |
| 仅支持核验 FGR（11 模型面板） | **22.98%** | 论文 Table 3 第二行 |

前两项是 `export_cf_cbr.py` 与 `eval_paper_tables.py` 的硬门；
§4.1 四项是 `render_paper_tables.py` 的硬门。

**重算已发表指标时不要自行添加条件**（例如排除退化框）。CBR 契约见根 `README.md`。

## 五、已知不一致

论文 PDF Table 3 第三行与本仓库复算**每项都不同，且本仓库更好**
（FGR 21.55 → 19.30、attribute CBR 10.21 → 8.50）。原因是 PDF 那行让探针头决定所有被
路由的行，导致跨类型池化污染。详见 `AUDIT.md` 第 6 条与 `reviewer_v060/CORRECTION.md`。
**论文正文需更新到新策略，或解释差异。**
