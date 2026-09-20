# MMDD QCPATH-R1 执行包

**这是一份下一轮实验规范，不是已经运行的实验或完整训练代码。**

建议先读主文档，再把整个目录交给Codex/执行模型。不要只给摘要或prompt；详细约束在主文档中。

| 文件 | 用途 |
|---|---|
| [EXPERIMENT_SPEC.zh-CN.md](EXPERIMENT_SPEC.zh-CN.md) | 唯一详细执行规范：数据、公式、A/B阶段、门槛、冻结范围、评测、产物、禁止事项 |
| [protocol.json](protocol.json) | 固定参数、执行顺序与门槛的机器可读副本 |
| [CODEX_PROMPT.md](CODEX_PROMPT.md) | 可直接提交给执行模型的指令 |
| [ACCEPTANCE_CHECKLIST.md](ACCEPTANCE_CHECKLIST.md) | H/A/B编号约束及合规回执要求 |
| [SOURCES_AND_BOUNDARIES.md](SOURCES_AND_BOUNDARIES.md) | 历史依据、本轮新增选择、读取与验证边界 |
| [reference/contracts.py](reference/contracts.py) | 数学参考内核；不是仓库训练器 |
| [reference/test_contracts.py](reference/test_contracts.py) | CPU合成语义测试 |
| [AUTHOR_VALIDATION.md](AUTHOR_VALIDATION.md) | 本交付实际完成的检查及未执行事项 |

## 当前授权

先运行A的E-only/QE两臂seed13；只有明确通过A门槛，才运行两臂B Teacher；A/B均通过才重复seed29。最多8个正式作业。KD、完整fresh、Stage2、200K、额外超参/seed均不执行。

本轮允许锁定健康B13/T0作局部试验parent，但新增训练组合从原始GT重建；不得依赖旧训练list。QT-only只能作对照，不能作为本轮主方法fallback。

## 本地CPU参考测试

```bash
cd MMDD_QCPATH_R1_20260919
python -m pytest reference/test_contracts.py -q
```

需要当前环境已有torch和pytest；不要为了运行此包擅自升级科研环境。真实数据、BASE/T0权重、原索引和仓库集成由执行者在服务器核验。本包没有提供假数据训练替身，也没有假装附带这些大文件。
