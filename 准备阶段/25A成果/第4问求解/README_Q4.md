# 问题 4 运行说明

本目录采用“单烟幕上下两圆周连续极值用于优化，多烟幕完整圆柱表面 max-min 用于终检”的双层模型。FY1、FY2、FY3 各自独立选择航向、速度、投放时刻和引信延迟，目标为三段严格完整遮蔽区间的并集长度。

## 直接复算当前最好候选

```powershell
python .\q4_strict_ring.py --mode verify
python .\verify_q4_outputs.py
python .\test_q4_model.py
```

生成：

- `result2.xlsx`：官方模板结果；
- `q4_runs/q4_best.json`：完整参数、区间和联合精检；
- `q4_timeline.png`：三架无人机遮蔽时间轴。

## 重新搜索

```powershell
python -u .\q4_strict_ring.py --mode quick 2>&1 | Tee-Object .\q4_runs\logs\quick_run.log
python -u .\q4_strict_ring.py --mode full 2>&1 | Tee-Object .\q4_runs\logs\full_run.log
```

搜索按 FY1/FY2/FY3 分阶段写入 `q4_runs/checkpoints`，每代状态写入 `q4_runs/logs/status.json`。中断后重跑会利用已有最佳候选；`--force` 强制重新搜索，`--skip-coop` 跳过较慢的完整表面联合精检。

当前连续圆周严格复算候选的三段窗口互不重叠，总并集约 `11.141794 s`。
