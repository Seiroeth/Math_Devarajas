# 问题2：最大恒定放热流量

本问读取问题1辨识的 `K=27973.462 W/K`、`UA=1798.827 W/K`，对3 h放热过程执行连续约束检查。

## 运行

```powershell
$env:MPLCONFIGDIR='C:\Users\35786\Desktop\文件\竞赛\数学建模\模拟题\.codex_work\mplconfig_q2'
& 'C:\ProgramData\Anaconda3\python.exe' -X utf8 '.\src\solve_q2.py'
```

## 结论

在题面参数链下，`q=0` 时3 h末顶部温度仅约409.725℃，已经违反 `T1>=515℃`，所以原问题的可行集为空，不存在满足全部约束的最大恒定放热流量。程序保留了二分搜索的括区间失败信息，并用两级高精度网格及标量最大裕量优化独立确认0—100 kg/s内无可行点。

`results/q2_final_result.json` 保存不可行性证书、参数恢复阈值和RK45交叉验证；`logs/run.log` 保存完整运行过程。
