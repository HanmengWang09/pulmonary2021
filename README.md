# Karamolegkos 2021 肺部模型复现

项目复现由四血池肺循环、单腔呼吸力学、血气与短期神经调节组成。独立闭环带有文献中的四腔泵和两床体循环，用于提供可运行的边界。模型的公式、参数来源和已验证指标见[复现记录](复现记录.md)。

| 入口 | 用途 |
| --- | --- |
| `pulmonary.py` | 图 3、表 II 的四段肺循环，边界给定右室流出、左房压力、胸膜压、肺泡压和肺泡容积 |
| `mechanical.py` | 单腔呼吸、四腔泵和体肺循环组成的机械闭环 |
| `gas_transport.py` | 肺泡和组织 O₂/CO₂ 交换、运输时延和按流向输运血气 |
| `closed_loop.py` | 自主呼吸、血气、压力反射和化学反射组成的闭环。也可运行图 4、图 6 的研究工况 |
| `respiration.py` | Albanese 2016 四段气道参考及原肺循环示例驱动 |

肺动脉、肺小动脉、毛细血管、肺静脉顺应性分别为 **0.76、1.93、3.87、25.37 mL/mmHg**，无应力血量为 **0、43.30、64.94、105.6 mL**。默认名义外周阻力为 **0.0894 mmHg·s/mL**，肺泡参考容积为 **2.25 L**，分流比例为 **1.7%**。闭环的肺和胸壁无应力容积、两床部分负荷和初态属于项目重建。Hering–Breuer 切换可传入阈值，原作者的血气阈值映射仍待核对。

## 运行

在本目录的 PowerShell 中运行。依赖已有 biomodel 环境中的 NumPy 和 SciPy。计算结果与缓存保存到项目 D 盘目录。

```powershell
$run = 'D:\data\OneDrive\Desktop\CardioPINN-Research-Agent\runs\pulmonary2021_desktop_update_20261124\runtime'
New-Item -ItemType Directory -Force -Path $run | Out-Null
$env:TEMP = $env:TMP = $env:MPLCONFIGDIR = $run
$env:PYTHONPYCACHEPREFIX = Join-Path $run 'pycache'
$python = 'D:\anaconda\envs\biomodel\python.exe'
& $python -m unittest discover -s tests -v
& $python pulmonary.py --duration-s 8 --output (Join-Path $run 'pulmonary.csv')
& $python closed_loop.py --case resting --duration-s 20 --output (Join-Path $run 'resting.csv')
```

直接调用 `pulmonary.simulate(boundary_at, duration_s)` 时，边界函数返回 `Boundary(Qrv, Pla, Ppl, PA, VA)`。压力单位为 mmHg，血流单位为 mL/s，肺泡容积单位为 L。

[Karamolegkos 2021 原文](https://doi.org/10.1109/OJEMB.2021.3128629) · [博士论文](https://doi.org/10.7916/D8H14JC3) · [项目参数对照](../CardioPINN-Research-Agent/runs/karamolegkos2021_full_reproduction_20261123/Karamolegkos2021与Munneke原PRef全量参数对照.md)
