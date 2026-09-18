# MMDD Stage1 CLEAN-R1 实验执行包

## 使用

把整个包交给仓库内的 Codex，发送 `CODEX_PROMPT.md`。主规范是 `EXPERIMENT_SPEC.zh-CN.md`，固定配置是 `clean_r1.json`。本轮只做原始带GT的20K湖上的Stage1：fresh Teacher → frozen Teacher → fresh Student SUP/KD → ANN + Teacher重排。

本包不是已经跑完的实验数据，也不是包含完整数据加载/训练CLI的产品代码。`reference_core.py`是可运行的模型/数学语义参考；生产数据管线与CLI需要按规范实现。原始湖与Qwen权重不在当前附件中。

## 文件

- `EXPERIMENT_SPEC.zh-CN.md`：完整步骤、公式、模型结构、标签、采样、loss、预算、评测、验收合同。
- `clean_r1.json`：唯一固定配置，不进行参数搜索。
- `CODEX_PROMPT.md`：执行指令。
- `reference_core.py` / `test_reference_core.py`：可直接运行的数学/模型参考与CPU测试。
- `REFERENCE_TEST_RECEIPT.txt`：当前环境实测回执，仅针对参考测试。
- `SOURCE_AUDIT.md`：当前源码字段/接口核对与hash。
- `RESOURCE_ESTIMATE.json`：按固定维度的解析字节估算与实际参考模型参数量，不是GPU延迟测试。
- `MANIFEST.sha256`：本包完整性。

CPU检查：

```bash
python -m unittest -v test_reference_core.py
```

依赖PyTorch。它不下载模型、不联网、不读历史checkpoint、不启动真实湖训练。真实执行需使用一张A100、本地已安装项目环境与原始数据。
