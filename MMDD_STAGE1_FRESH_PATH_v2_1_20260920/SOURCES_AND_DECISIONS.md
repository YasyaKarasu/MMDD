# 本版来源与证据边界

## 直接依据

当前用户明确要求独立QT-only对照、Student无evidence/旧模型对照、完整fresh不退回历史parent；纠正硬件为始终2×RTX4090，并允许同卡多任务和双卡独立分支并行。

实际读取了v2.0完整主文档、JSON、提示词、参考代码和测试；v2.0 §12.5明确只有T_PATH权重f0读出，不是独立QT训练；§9.1仅SUP-QE/KD-QE/KD-EONLY，不含独立Direct-only训练。新版本补齐而不将旧读出改名。

实际读取上传Stage1 bridge源码的 `src/mmdd_stage1/b13_recipe.py`，确认旧式五关系P/R、SUP 10×sigmoid、raw KD、不同学习率等。新KD-NATIVE只保持旧式结构，明确不声称精确复刻该完整配方。本版未重新审核全部旧训练运行，也未重新测历史checkpoint分数。

## 用户观察与新配置

训练/推理GPU利用率偏低是用户提供的观察；本版设置profile后的有限共驻，不声称本地测得4090吞吐提升。2任务/卡、85%显存预约、3GiB余量、1.05最低吞吐改善等是调度工作点，不是模型性能结论。

T_QT target训练、Direct Student全分母训练、KD-NATIVE的C2定义和新增repeat门槛均是本次明确新增的实验安排，尚未验证能接近历史最强。其余主线数学配方沿用v2.0，不静默修正或增加新研究方向。

## 实际执行边界

本次创建/检查了文档、配置、对照与调度参考代码，并执行CPU合成测试。没有访问用户GPU运行环境，没有正式Qwen前向，没有运行真实Teacher/Student训练或Recall评测。实际通过数量由测试回执记录，不沿用旧包的计数冒称新验收。

## 读取的输入文件SHA256

| 输入文件 | SHA256 |
|---|---|
| `MMDD_STAGE1_FRESH_PATH_20260920.zip` | `c15c4e0eefea415271bcbe0296e78a42ee45a4ce9decca754d09690a5e1c7757` |
| `EXPERIMENT_SPEC.zh-CN.md` | `b80ca32804486dbcedbe63e11b491b0ace0d56a0b21938bb9b4c18837b4449b9` |
| `MMDD_src_stage1_bridge_20260915.tar.gz` | `5a81896f62be4433eccb0cdb35552e69fdd569dcc335ae0b7adc447033b5c82a` |
