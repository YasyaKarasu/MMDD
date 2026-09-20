# 双 RTX 4090 并行执行合同

本文件细化主文档§4.6/13.5，不修改数学模型、训练预算和loss。

## 可以并行的工作

- T_EDGE完成且公共刷新列表/graph封存：T_PATH和T_QT分别在两张卡运行。
- S_SUP_C1不需要Teacher软标签，候选准备好即可与其它无依赖作业并行；S_KD_C1等待T_PATH冻结后启动。
- QT-SUP/C1和C2不需要任何Teacher logits；QT-KD等待T_QT。两个Student分支可分别占卡。
- KD-C1完成后，KD-QE、KD-EONLY同时启动在两卡，KD-NATIVE还需T_QT；有足够资源可同卡共驻。
- 纯CPU候选/索引元数据准备及独立模型的评分可与GPU训练重叠，但不得读取未发布权重/缓存。

## 不可以并行成“共享训练”的工作

同一模型的连续epoch/阶段不能并行更新；两个进程不能写同一个optimizer/state文件；Teacher还在更新时不能给Student提供未固定版本的KD。两个Student不能一起累积到同一个optimizer，不用DDP把独立对照误做一个分布式模型。

## 物理/逻辑设备示例

以下命令是要求实现的CLI接口示例，不表示服务器已有程序。只有上游完成回执齐全才执行：

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_stage1_fresh_path.py train-teacher \
  --seed 13 --stage path --device cuda:0 --protocol protocol.json \
  --work-dir work/mmdd_stage1_fresh_path_v2_1_20260920 > path13.log 2>&1 &
pid_path=$!
CUDA_VISIBLE_DEVICES=1 python src/run_stage1_fresh_path.py train-teacher \
  --seed 13 --stage qt --device cuda:0 --protocol protocol.json \
  --work-dir work/mmdd_stage1_fresh_path_v2_1_20260920 > qt13.log 2>&1 &
pid_qt=$!
wait "$pid_path"; rc_path=$?
wait "$pid_qt"; rc_qt=$?
printf 'path=%s qt=%s\n' "$rc_path" "$rc_qt"
test "$rc_path" -eq 0 && test "$rc_qt" -eq 0
```

`run --devices 0,1`应管理依赖/退出码，无需用户手动提交每一个命令。每个子进程从共享run配置解析root路径，并在各自分支目录写状态；不能两个进程都把CUDA_VISIBLE_DEVICES设置为0却记录不同物理设备。

## 同卡共驻门槛

默认一卡1作业，允许一卡2作业/全机最多4。先profile，不修改正式batch：2个warm-up+8个测量update；预约显存=max(框架reserved峰值,可测进程占用峰值)+1GiB。加外部占用后≤设备总量85%，且≥3GiB空闲。固定工作量共驻相对独占依次执行的makespan加速≥1.05，数值差分通过后才共驻。

profile使用临时克隆、独立RNG和固定样本，结果权重不接入正式训练。调度事件记录配置/指纹、物理UUID、PID、开始/结束、各阶段实际重算开销。预取最多4worker和LRU最多8GiB是全机总数。

Qwen补缓存阶段默认独占所在卡，另一卡可跑无需缺项的任务。任何任务在独占也超资源时按主文档处理；不能砍样本/模态/路径让它“通过”。

## 崩溃、OOM与测量

OOM退出保留最后完整提交状态，丢弃失败step部分梯度，再以相同配置独占重放；不换seed、AMP或logical batch。若可用显存受其它非本实验进程影响，保留资源记录、延迟受影响任务，不停止其它就绪独立分支。

所有正式单请求时延p50/p95使用无其它实验作业的GPU窗口；并发吞吐单独报告。完成耗时用整次run真实wall time；累计GPU任务时间可另报，但不要把重叠墙钟相加。

共享纯缓存只读；需要补充时由单writer锁+临时文件+校验+原子发布。每条score cache key必须有Teacher分支和checkpoint；相同ID并不代表T_QT与T_PATH分数可以复用。
