# 2025 A 题第 5 问：严格圆柱全遮蔽求解

本目录把第 5 问拆成“配对探针—分派—同机多弹精修—严格终算”四层。所有时长均按真目标圆柱完整遮蔽计算，不使用目标中心点近似。

## 当前结果

| 导弹 | 严格完整遮蔽并集 |
|---|---:|
| M1 | 7.6190376709 s |
| M2 | 10.7124720662 s |
| M3 | 3.7215731490 s |
| 合计 | **22.0530828861 s** |

以 1°/0.5 m 高密度完整表面采样进行多烟幕协同复核，结果为 22.0531003870 s，与逐烟幕严格并集只差约 0.000018 s，说明本方案收益主要来自时间窗拼接，不依赖粗网格“擦边”。

分派为 M1←FY1（三弹），M2←FY2（双弹）+FY3+FY4，M3←FY5。共使用 8 枚策略弹；同一无人机的航向和速度完全一致，相邻投弹间隔均不少于 1 s。

## 运行

在本目录执行：

```powershell
python q5_strict_ring.py --mode verify --cooperative
```

该命令约数秒完成严格复算，生成：

- `result3.xlsx`：在官方模板上填写的最终答案；
- `q5_runs/q5_best.json`：所有决策变量、投放点、起爆点、严格区间；
- `q5_timeline.png`：三枚导弹的遮蔽时间轴；
- `q5_runs/status.json` 与 `q5_runs/q5_run.log`：状态和运行日志。

从当前种子继续搜索：

```powershell
python q5_strict_ring.py --mode optimize --iterations 8 --cooperative
```

运行 15 个 UAV–导弹单弹配对探针：

```powershell
python q5_strict_ring.py --mode probe --iterations 12
```

探针按 `q5_runs/probes/FYx_Mk.json` 独立落盘。再次运行会跳过已完成组合；加 `--force` 才会重算。优化阶段每完成一条无人机航线也会刷新 `q5_best.json` 和 `status.json`，可用 `--resume` 从检查点恢复。

高密度完整表面复核可使用：

```powershell
python q5_strict_ring.py --mode verify --fine-cooperative
```

## 验证

```powershell
python test_q5_model.py
python verify_q5_outputs.py
```

第一条检查三枚导弹命中时刻、策略约束和严格数值基准；第二条从 JSON 重新构造 8 枚烟幕弹，独立复算三导弹时长，并逐行核对官方 `result3.xlsx` 模板、坐标、时长和导弹编号。

## 模型口径

对每个时刻、真目标上下圆周的所有连续角度，计算烟幕球心到“导弹—目标点视线线段”的最大距离。最大距离不超过 10 m 时，才认定整根圆柱被一枚烟幕完整遮蔽。优化阶段用较密角度网格快速评分；最终结果用连续角度极值和 Brent 根定位时间边界。M1、M2、M3 分别求时间并集，目标函数是三个并集时长之和。

公共严格几何代码位于 `../第3问求解/smoke_strict_core.py`。脚本按自身位置自动寻找项目根目录和官方模板，所以从本目录、上级目录或绝对路径启动都不会依赖当前工作目录。
