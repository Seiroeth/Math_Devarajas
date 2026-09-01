# 问题3：多目标放热控制

本目录严格读取问题1辨识参数，运行48段控制、加权和SLSQP、多目标进化和分段需求反演。计算证明原约束可行域为空；所有非零控制结果均作为诊断方案保存，不作为可行推荐。

运行：在工作目录执行 `C:\ProgramData\Anaconda3\python.exe -X utf8 模拟题结果\问题3_多目标放热控制\src\solve_q3.py`。若Matplotlib缓存无写权限，请把 `MPLCONFIGDIR` 指向工作目录内可写文件夹。

- `config`：参数、时间步长和随机种子。
- `results`：算法运行表、10 s轨迹和不可行性证书。
- `figures`：论文图。
- `logs`：完整配置与运行结果。
- `report_问题3.docx`：完整推导、算法比较和结论。
