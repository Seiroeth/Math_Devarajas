# 问题1：参数辨识与温度预测

## 任务

使用题目表1中 0—6 h 的 T1、T3、T5 共21个观测量辨识层间等效换热参数 K 和每层散热参数 UA；比较三类辨识算法和三种ODE求解器；计算新工况0—4 h每60 s的五层温度，并填写根目录 `result1.xlsx` 的 B:G 列。

## 环境

- Windows PowerShell
- Python：`C:\ProgramData\Anaconda3\python.exe`
- 主要依赖：NumPy 1.24.4、SciPy 1.9.1、pandas 1.4.4、Matplotlib 3.5.2
- 随机种子：20260825、20260826、20260827

## 运行

在本文件夹执行：

```powershell
$env:MPLCONFIGDIR='C:\Users\35786\Desktop\文件\竞赛\数学建模\模拟题\.codex_work\mplconfig_q1'
& 'C:\ProgramData\Anaconda3\python.exe' -X utf8 '.\src\solve_q1.py'
```

程序读取 `config/q1_config.json`，调用根目录 `common/five_layer_model.py`，并覆盖生成 `data`、`figures`、`results` 和 `logs` 中的问题1结果。Excel写入由独立的工作簿脚本执行，原模板备份为 `data/result1_original_backup.xlsx`。

## 主要输出

- `report_问题1.docx`：完整推导、算法比较、误差、置信区间、灵敏度和预测报告；
- `results/identified_parameters.json`：供后续四问读取的统一参数文件；
- `results/prediction_60s.csv`：0—4 h共241个时刻的T1—T5和底部出口温度；
- `results/algorithm_comparison_runs.csv`：三类算法、三次独立运行的原始比较结果；
- `results/ode_solver_comparison.csv`：矩阵指数、RK45和RK4交叉验证；
- `results/sensitivity_analysis.csv`：K、UA分别±10%的局部灵敏度；
- `logs/run.log`：运行参数、收敛结果和验收量。

## 验收标准

1. 表1全部21个观测量参与目标函数；
2. 最终最大绝对残差不超过0.6℃；
3. 三类算法收敛到相同参数邻域；
4. RK45、RK4与矩阵指数结果在声明精度内一致；
5. 新工况输出恰有241行，时间从0到14400 s、步长60 s；
6. `result1.xlsx` 只改变B4:G244，且G列逐行等于F列；
7. 总能量守恒残差接近机器精度。
