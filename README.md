# baseline代码位置说明

| Baseline    | 代码位置                                             | 说明                                                         |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| DLinear     | /home/lzm/Time-Series-Library/models/DLinear.py      | 集成在 thuml 官方 Time-Series-Library 框架里                 |
| PatchTST    | /home/lzm/Time-Series-Library/models/PatchTST.py     | 同上                                                         |
| iTransforme | /home/lzm/Time-Series-Library/models/iTransformer.py | 同上                                                         |
| TimeMixer   | /home/lzm/Time-Series-Library/models/iTransformer.py | 同上                                                         |
| RAFT        | /home/lzm/RAFT/models/RAFT.py                        | 两处都有，/home/lzm/RAFT 里的 checkpoints/results 是软链接指到Time-Series-Library |

  权重位置：统一放在 /home/lzm/Time-Series-Library/checkpoints/，按 long_term_forecast_{数据集}_{seq}_{pred}_{模型}_..._baseline_0/checkpoint.pth 命名，共 161 个目录，总计 2.1G：

