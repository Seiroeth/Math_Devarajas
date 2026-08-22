# 问题 3 运行说明

代码沿用 Q1/Q2 的“完整圆柱 + 上下两圆周连续极值”判据。优化目标是三枚烟幕各自完整遮蔽区间的并集长度，三枚弹共享 FY1 的航向和速度，投放时刻按 `td2=td1+1+s2`、`td3=td2+1+s3` 参数化。

## 运行

在本目录打开 PowerShell：

```powershell
python .\q3_strict_ring.py --mode verify
python .\verify_q3_outputs.py
python .\test_strict_model.py
```

`verify` 对当前最好候选做严格复算并生成：

- `result1.xlsx`：按赛题官方模板填写；
- `q3_runs/q3_best.json`：包含投放/起爆时刻、点坐标、严格区间与联合精检；
- `q3_timeline.png`：三枚弹和总并集的时间轴图。

重新搜索：

```powershell
python -u .\q3_strict_ring.py --mode quick 2>&1 | Tee-Object .\q3_runs\logs\quick_run.log
python -u .\q3_strict_ring.py --mode full 2>&1 | Tee-Object .\q3_runs\logs\full_run.log
```

每一代会更新 `q3_runs/logs/status.json` 和 `q3_runs/checkpoints/*_latest.json`。中断后重跑会把已有最佳解作为新种群起点；完整阶段已完成时会直接读取检查点。使用 `--force` 可强制重跑。

`--skip-coop` 可跳过较慢的完整圆柱表面 `max-min` 多烟幕协同精检；独立圆周严格复算始终执行。

当前严格复算候选为：航向 `179.65°`、速度 `140 m/s`，三弹投放时刻约为 `0`、`3.716756`、`5.588584 s`，独立完整遮蔽并集约为 `7.619038 s`。高密度表面 max-min 精检的额外协同增益约为 `9e-7 s`，可视为主要由时间接力贡献。
