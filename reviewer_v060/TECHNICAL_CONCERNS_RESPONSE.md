# GroundBind 审稿人技术关切回应

本文件回应审稿人提出的三组技术局限与方法问题。

---

## 1. 图像交换 vs 绑定正确性；无框优于有框的解释调整

### 审稿人关切
> 图像交换证明的是评分对图像输入具有敏感性，而不是已经识别候选实例的绑定正确性。尤其是，无框的 image-level 提示在当前诊断中优于 boxed-region 提示，这使"候选区域级验证是关键能力来源"的解释需要进一步调整。

### 回应

**图像交换的主张范围已限定**：论文 §5.2 和 Appendix 图像交换章节的措辞为"验证器评分对**图像输入敏感**"（image-grounded），而非"已识别候选实例的绑定正确性"。我们的论证链是：
1. 图像交换（Δ=11.41 显著）→ 验证器真正使用图像像素，非纯文本先验
2. 去框消融（AUROC 0.5923→0.6294，0/13 下降）→ 判别力来自**图像-表达全局一致性**，非候选框定位
3. 结论（论文 §5.2 末尾已调整）："验证器的判别力来源于图像-文本一致性校验，而非区域级定位能力"

**"无框优于有框"不削弱主张，而是精确界定能力边界**：
- 论文核心主张（§1 / §5）：ASV 通过**后验一致性校验**缓释幻觉，而非重新定位
- 去框现象**支持**这一主张——如果验证器靠重新定位候选区域，去框应导致判别力塌陷；实测不塌反升，说明它走的正是全局一致性路径
- 能力来源不是"候选区域级验证"，而是"图像-文本对齐校验"——这正是我们在 §C.9 机制分析中阐述的设计

**调整建议**（论文侧）：
1. §5.2 消融段落明确写"验证器判别力来自 image-grounded consistency，非 region-localized re-positioning"
2. Appendix 去框章节标题改为"No-Box Ablation: Verification Beyond Bounding Boxes"
3. 删除任何暗示"候选区域级验证是关键能力来源"的表述（如有），统一口径为"图像-文本一致性校验"

---

## 2. Direct 与 Joint 接受头标签口径不一致

### 审稿人关切
> Direct 接受头仍使用表达正负标签，joint 接受头则使用包含 IoU 条件的候选正确性标签，两者尚未统一。

### 回应与澄清

**两套接受头的标签口径确实不同，但这是任务定义差异的必然结果，非设计缺陷**：

#### Direct 接受头（OmniVerifier 路由）
- **任务**：判断"表达本身在图像中是否可满足"（与候选框无关的二分类）
- **标签**：从 benchmark 的 `htype` 字段直接读取
  - `positive` 表达（图像支持）→ label=1
  - `object/co_occurrence/attribute/relation` 幻觉（图像不支持）→ label=0
- **输入**：表达文本 + 图像（或候选框标红的图像）
- **代码**：`eval_tasks_t1_t3_t4.py` L112-114 直接用 `htype == 'positive'` 为 label

#### Joint 接受头（JEV-2B 路由）
- **任务**：判断"**候选框是否正确定位了表达所指实例**"（候选绑定正确性，IoU-conditioned）
- **标签**：positive 表达 + IoU≥0.5 → 1；其余 → 0
  - 原因：只有候选**既定位了真实物体**（IoU≥0.5）**又该物体匹配表达**（positive）时，绑定才正确
  - 幻觉表达（图像不支持）即使 IoU 高也不是"正确绑定"→ label=0
- **输入**：候选文本 + 图像 + 空间关系上下文（joint prompt）
- **代码**：`eval_tasks_t1_t3_t4.py` L297 `label = 1 if (is_pos and iou >= 0.5) else 0`

#### 为什么不能统一
两个头回答**不同的问题**：
- Direct："这个表达图像支持吗？"（与候选框质量无关）
- Joint："这个候选框正确定位了表达所指吗？"（IoU 是绑定正确性的必要条件）

统一为"表达正负"会让 joint 头失去候选质量信号；统一为"IoU 条件正确性"会让 direct 头无法在无框场景下工作（如去框消融、无框下游任务）。

**论文侧建议**：
1. §C.6（direct 头）/ §C.9（joint 头）明确写出各自的标签定义与任务差异
2. 增加一段（§C.10 或 discussion）："两套接受头的标签口径不同，反映其任务定义差异——direct 头判断表达可满足性，joint 头判断候选绑定正确性（IoU-conditioned）。这使 ASV 能适配无框下游任务（direct）与有框诊断场景（joint）。"

---

## 3. 串联校准未匹配实际保留率

### 审稿人关切
> 串联校准实验有帮助，但它仍然没有匹配实际正确定位保留率。

### 现状与回应

**Q4 串联校准已完成，但"匹配实际保留率"是指什么？**
- 如果指"serial 阈值校准后的 positive 保留率应等于 support 的保留率"：
  - 当前 Q4 实验（`refit_struct.json`）是**串联校准**（先在 support 上 fit 到 target=0.95，再用该阈值跑 full pipeline）
  - serial 的 positive 保留率（coordfix 后）为 **94.72%**，support 为 **95.00%**（target），差 0.28pp
  - 不完全相等是因为结构分支的候选分布与 support 不同（relation 幻觉的竞争探针分数偏移）

- 如果指"串联校准的结果应与论文主表 Table 2/3 一致"：
  - 论文主表用的是**并联校准**（full pipeline 直接 fit 到 0.95，各模型独立阈值）
  - Q4 是审稿补充实验，证明"即使用 support 的阈值（串联），结构分支仍优于 support"
  - 两者口径不同是 by design（对照实验），不是缺陷

**如果审稿人要求"论文主表应改用串联校准"**：
- 可行，但需重新跑 13 模型 × matched eval（约 6-8 小时 GPU 时间）
- 建议先确认审稿人具体诉求——如果只是"展示串联校准结果"，Q4 已满足；如果要求"主表改用串联"，需另排期

**论文侧建议**（假设不改主表）：
1. §5.1 / Appendix Q4 章节明确写"串联校准口径：阈值在 support 上 fit 到 target=0.95，然后应用于 full pipeline（包括结构分支）"
2. 报告 serial 的实际 positive 保留率（94.72%）与 support（95.00%）的小偏差，并解释原因（分布偏移）
3. 如果审稿人坚持"主表必须串联校准"，回复时说明重跑成本与时间，询问是否 required change

---

## 4. Parse-fail 默认 NO 对 FA/BA 的污染（已实质修正）

### 审稿人关切
> UniVG-R1 和 TreeVGR 的大量 verification 解析失败被默认成 NO，并进入 FA/BA 计算。这影响"语言判断与定位行为差异"的解释，不能仅靠在附录披露便视为解决。

### 已完成的源头实质修正

**修正口径**（commit `95a32c2`，2026-10-07）：
- **t1（discriminative VQA）表现改为 abstention 口径**：FA/TPR/BA 只在 `parse_valid=True` 子集上计算
- **解析失败单列为 abstention_pos / abstention_neg / abstention_all**，不塞进 YES/NO 默认
- **旧字段重命名为 `*_legacy_defaultNO`**，保留审计可追溯性但不作主表

**污染量化**（`abstention.json` / `audit_parse_impact.json`）：

| 模型 | pos 失败率 | neg 失败率 | 旧 BA（污染） | 新 BA（abstention） | ΔBA |
|---|---|---|---|---|---|
| **UniVG-R1** | 46.2% | 35.0% | 60.42% | **73.08%** | **+12.66pp** |
| **TreeVGR** | 0.8% | 30.9% | 76.60% | **66.78%** | **−9.82pp** |
| 其余 11 模型 | ~0% | ~0% | — | — | ≈0 |

- UniVG-R1 正例高失败率 → 旧口径虚低 BA（假 NO = 错误拒绝）
- TreeVGR 负例高失败率 → 旧口径虚高 BA（假 NO = 正确拒绝）
- 两者方向相反，污染既非常数也非单向，无法靠统一声明抵消

**对"语言判断 vs 定位行为差异"论点的影响——幸存且更强**：
- 修正后 TreeVGR 从第 8 跌至最后，UniVG-R1 从垫底升至第 10，两者位置对调
- **高弃权率本身就是"语言判断不可靠"的直接证据**——这两个模型定位通道正常输出框，语言判断通道大量解析失败，正是"双通道脱节"的极端案例
- 论点不削弱，反而更干净（不依赖被污染的 FA/BA）

**论文侧对齐建议**：
1. **Table 1（t1 主表）改用 abstention 口径**：FA/TPR/BA 列脚注写"computed on parse_valid subset; abstention rates in Appendix A.1"
2. **Appendix A.1 从"披露"改"修正"**：
   - 标题：~~"E1 Limitation: Parsing Heterogeneity"~~ → "Abstention-Aware Evaluation: Parsing Failure Handling"
   - 正文：删除"recorded separately as a known limitation"，改"We recompute FA/TPR/BA on the parse_valid subset to avoid default-NO contamination. Two models (UniVG-R1, TreeVGR) show high abstention rates (37.24%, 24.84%), reflecting unreliable linguistic judgment — a direct manifestation of the language-vs-localization discrepancy."
3. **§5.1 讨论段**：提及高弃权率强化而非削弱主论点

**交付清单**（已推送 `exp/reviewer-v060-supplement` 分支 `95a32c2`）：
- 源头修正脚本：`eval_tasks_t1_t3_t4.py`（eval_t1 函数 + 表格列 abstain%）
- 审计脚本：`audit_parse_impact.py`（三口径对比）+ `eval_abstention.py`（统计）
- 结果：`abstention.json` / `audit_parse_impact.json` / `tasks_t1_t4.json`（新口径全量）
- 文档：`REVIEWER_RESPONSE.md` Q8 章节已改写为"实质修正"而非"附录披露"

---

## 总结：三类关切的状态

| 关切 | 状态 | 交付物 | 论文侧行动 |
|---|---|---|---|
| **实验协议不清晰** | ✅ 已完成 | `EXPERIMENTAL_PROTOCOLS.md` | 引用本文件或吸收到 Appendix |
| **图像交换 vs 绑定正确性；无框解释** | ✅ 已澄清 | 本文 §1 | §5.2 / Appendix 措辞调整 |
| **Direct/Joint 标签口径不一致** | ✅ 已说明（by design） | 本文 §2 | §C.6/C.9/C.10 补充说明 |
| **串联校准未匹配保留率** | ⚠️ 待明确诉求 | Q4 已有串联校准 | 确认是否需改主表 |
| **Parse-fail 污染 FA/BA** | ✅ 已实质修正 | abstention 口径 + 审计 | Table 1 改用新口径；A.1 改"修正" |

**下一步**：
1. 论文侧吸收 `EXPERIMENTAL_PROTOCOLS.md` 内容（Appendix 或 supplementary material）
2. 按本文 §1-4 建议调整论文措辞（§5.2 / Appendix A.1 / §C.6-10）
3. 确认审稿人对"串联校准"的具体要求（主表是否必须改）
4. 提交前：仓库凭据扫描、README 更新 GroundBind 命名、推送 `exp/reviewer-rebuttal` 分支
