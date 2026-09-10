# 测试

`src/` 的研究流程和仍用于实际数据集构造的 `scripts_old/` 都有测试。
不能仅凭目录名或实验轮次判断测试已废弃：旧轮次的部分函数仍被后续实验复用。

使用 `MMDD` 环境。日常修改优先运行对应文件，例如：

```bash
# WDC 采样
conda run -n MMDD python -m pytest tests/test_wdc200k_sampling.py -q

# 当前数据集流程
conda run -n MMDD python -m pytest tests/test_src_dataset_builder.py tests/test_wdc_streaming_builder.py -q

# Stage 1 模型、训练控制和路径聚合
conda run -n MMDD python -m pytest tests/test_stage1_models.py tests/test_stage1_training_control.py tests/test_stage1_r6_retrieval.py -q

# 全套回归并查看实际慢点
conda run -n MMDD python -m pytest tests -q --durations=20
```

`pytest.ini` 统一配置导入路径，单独运行文件不再依赖其他测试先修改
`sys.path`。默认仍运行所有测试，没有隐藏或跳过慢测试。

`conftest.py` 将测试工作目录切换到临时目录，避免默认配置查找接触仓库内的
用户文件。配置相关测试使用临时合成配置；不要读取仓库密钥或真实服务配置。
少量网络协议测试启动本机回环服务，需要创建本地 socket 的权限，但不需要外网、
真实模型服务或 GPU。

清理重复测试时，应确认实际输入和执行路径相同。参数化若没有改变被测输入，
应合并；采样、恢复、并发、数据完整性和证据机制等行为断言应保留。
