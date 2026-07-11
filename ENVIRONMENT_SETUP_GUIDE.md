# MORPH 环境配置指南

## 快速开始

### 方法一：自动配置（推荐）

1. **运行配置脚本**
   ```bash
   cd MORPH
   setup_environment.bat
   ```

2. **等待安装完成**（约5-10分钟）

3. **在VSCode中打开Jupyter Notebook**
   - 打开 `Step1_Contextualization_Scaling/Test/1.1_test_Convert_TrackingData.ipynb`
   - 点击右上角选择内核
   - 选择 `Python (MORPH)`

4. **开始运行**

### 方法二：手动配置

#### 步骤1：创建虚拟环境

```bash
cd MORPH
python -m venv MORPHenv
```

#### 步骤2：激活虚拟环境

**Windows:**
```bash
MORPHenv\Scripts\activate
```

**Linux/Mac:**
```bash
source MORPHenv/bin/activate
```

#### 步骤3：升级pip

```bash
python -m pip install --upgrade pip
```

#### 步骤4：安装依赖包

```bash
pip install -r requirements.txt
```

#### 步骤5：安装unravelsports（可选但推荐）

⚠️ **重要提示**: 必须使用modified版本，原版存在严重的依赖冲突！

**依赖冲突说明**：
- **原版问题**: `keras==2.14.0` (严格锁定) 与 TensorFlow 2.18+ 要求的 `keras>=3.5.0` 冲突
- **Modified版本**: 已将 `keras==2.14.0` 改为 `keras>=2.14.0`，解决冲突
- **错误信息**: 如果看到 `ResolutionImpossible` 或 `conflicting dependencies`，说明使用了错误的版本

**正确安装方法**：

```bash
# 设置UTF-8编码（解决Windows GBK编码错误）
set PYTHONUTF8=1

# 从本地modified版本安装（已修复依赖冲突）
pip install "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"
```

**验证安装**：
```bash
python -c "from unravel.soccer import KloppyPolarsDataset; print('unravelsports安装成功！')"
```

**常见错误**：

1. **UnicodeDecodeError: 'gbk' codec** → 确保已执行 `set PYTHONUTF8=1`
2. **ResolutionImpossible (keras冲突)** → 必须使用modified版本路径
3. **找不到文件夹** → 路径包含空格，必须用引号括起来

**如果仍然失败**：
- 跳过此步骤，直接进行步骤6
- unravelsports不是必需的（可以直接使用kloppy）
- 详见下方"Q2: unravelsports安装问题"和"Q7: unravelsports依赖冲突"

#### 步骤6：配置Jupyter内核

```bash
python -m ipykernel install --user --name=MORPHenv --display-name="Python (MORPH)"
```

## 验证安装

运行以下Python代码验证关键包是否正确安装：

```python
import sys
print(f"Python版本: {sys.version}")

# 核心包
import numpy as np
import pandas as pd
import polars as pl
print(f"✓ NumPy {np.__version__}")
print(f"✓ Pandas {pd.__version__}")
print(f"✓ Polars {pl.__version__}")

# 图神经网络
import torch
import torch_geometric
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ PyTorch Geometric {torch_geometric.__version__}")

# 追踪数据处理
import kloppy
from unravel.soccer import KloppyPolarsDataset
print(f"✓ Kloppy {kloppy.__version__}")
print(f"✓ Unravelsports 已安装")

# 贝叶斯推断
import pyro
import numpyro
import arviz
print(f"✓ Pyro {pyro.__version__}")
print(f"✓ NumPyro {numpyro.__version__}")
print(f"✓ ArviZ {arviz.__version__}")

# 可视化
import matplotlib
import seaborn as sns
import mplsoccer
print(f"✓ Matplotlib {matplotlib.__version__}")
print(f"✓ Seaborn {sns.__version__}")
print(f"✓ mplsoccer {mplsoccer.__version__}")

print("\n所有核心包安装成功！")
```

## GPU支持（可选）

如果你有NVIDIA GPU并希望加速训练，请根据CUDA版本重新安装PyTorch：

### CUDA 11.8
```bash
pip uninstall torch torch-geometric
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### CUDA 12.1
```bash
pip uninstall torch torch-geometric
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
```

### 验证GPU
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

## VSCode配置

### 1. 安装必要的VSCode扩展

- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **Pylance** (Microsoft)

### 2. 选择Python解释器

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择 `MORPHenv` 虚拟环境中的Python

### 3. 配置Jupyter内核

1. 打开任意 `.ipynb` 文件
2. 点击右上角的内核选择器
3. 选择 `Python (MORPH)`

## 常见问题

### Q1: pip install 速度很慢

**解决方案**：使用国内镜像源

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: unravelsports安装失败

**症状1：网络超时**
```
Cloning https://github.com/UnravelSports/unravelsports.git
(一直卡住)
```

**解决方案**：使用本地modified版本
```bash
set PYTHONUTF8=1
pip install "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"
```

**症状2：编码错误**
```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xbd
```

**解决方案**：设置UTF-8编码
```bash
set PYTHONUTF8=1
pip install "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"
```

**症状3：依赖冲突**
```
ERROR: ResolutionImpossible
The conflict is caused by:
    unravelsports 1.1.0 depends on keras==2.14.0
    tensorflow 2.18.1 depends on keras>=3.5.0
```

**解决方案**：参见Q7详细说明

### Q3: torch-geometric安装失败

**解决方案**：按顺序安装依赖

```bash
pip install torch
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### Q4: Jupyter内核未显示

**解决方案**：重新配置内核

```bash
# 激活虚拟环境
MORPHenv\Scripts\activate

# 重新安装内核
python -m ipykernel install --user --name=MORPHenv --display-name="Python (MORPH)" --force

# 重启VSCode
```

### Q8: Jupyter内核连接超时 🔴 新问题！

**症状**：
```
连接到kernel：MORPHenv (python 3.12.4)
(一直显示连接中，但无法连接成功)
```

**快速解决方案**：

**步骤1：完全关闭VSCode**
- 点击右上角 ❌ 关闭所有窗口
- 或按 `Alt+F4`
- 确保任务管理器中没有VSCode进程

**步骤2：重新打开VSCode**
- 双击VSCode图标启动

**步骤3：打开notebook并选择内核**
- 打开 `1.1_test_Convert_TrackingData.ipynb`
- 点击右上角内核选择器
- 选择 `Python (MORPH)` 或 `morphenv`
- 等待5-10秒连接

**步骤4：验证连接**
- 运行第一个单元格
- 如果能看到输出，说明成功

**如果仍然失败**：

**方案A：重新加载窗口**
```
1. 按 Ctrl+Shift+P
2. 输入 "Reload Window"
3. 选择 "Developer: Reload Window"
4. 重新选择内核
```

**方案B：清除Jupyter缓存**
```bash
# 停止所有Jupyter进程
taskkill /F /IM jupyter.exe /T 2>nul
taskkill /F /IM python.exe /T 2>nul

# 删除运行时文件
rmdir /s /q %APPDATA%\jupyter\runtime

# 重新注册内核
cd E:\JerryWu\Master\SoccerAnalytics\G-TAF\MORPH
MORPHenv\Scripts\activate
python -m ipykernel install --user --name=MORPHenv --display-name="Python (MORPH)" --force

# 重启VSCode
```

**方案C：使用命令行测试**
```bash
# 激活环境
cd E:\JerryWu\Master\SoccerAnalytics\G-TAF\MORPH
MORPHenv\Scripts\activate

# 启动Jupyter
jupyter notebook

# 在浏览器中测试内核是否正常
```

**常见原因**：
- VSCode缓存了旧的内核信息
- Jupyter服务器启动失败
- 端口被占用
- 防火墙阻止连接

详细诊断和解决方案请参考 [`MORPH_ADAPTATION_GUIDE.md`](MORPH_ADAPTATION_GUIDE.md) 的Q11部分

### Q5: 导入kloppy或unravelsports失败

**解决方案**：检查版本兼容性

```bash
# 卸载并重新安装
pip uninstall kloppy unravelsports -y
pip install kloppy>=3.17.0
pip install git+https://github.com/UnravelSports/unravelsports.git
```

### Q6: 内存不足

**解决方案**：
- 关闭其他程序释放内存
- 使用测试版notebook（单场比赛）而非通用版（64场比赛）
- 分批处理数据

### Q7: unravelsports依赖冲突 🔴 重要！

**症状**：
```
ERROR: Cannot install unravelsports because these package versions have conflicting dependencies.

The conflict is caused by:
    unravelsports 1.1.0 depends on keras==2.14.0
    tensorflow 2.20.0 depends on keras>=3.10.0
    tensorflow 2.19.1 depends on keras>=3.5.0

ERROR: ResolutionImpossible
```

**问题诊断**：

这是一个**硬性依赖冲突**：

| 版本 | keras要求 | 状态 |
|------|-----------|------|
| 原版unravelsports | `keras==2.14.0` | ❌ 与新版TensorFlow冲突 |
| Modified版本 | `keras>=2.14.0` | ✅ 兼容新版TensorFlow |

**验证问题来源**：
```bash
# 查看原版依赖
cd E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main
type setup.py | findstr /i "keras"
# 输出: "keras==2.14.0"  ← 严格锁定

# 查看modified版本依赖
cd "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"
type setup.py | findstr /i "keras"
# 输出: "keras>=2.14.0"  ← 允许更新
```

**解决方案A：使用modified版本（强烈推荐）**

```bash
# 1. 确保在激活的虚拟环境中
MORPHenv\Scripts\activate

# 2. 设置UTF-8编码
set PYTHONUTF8=1

# 3. 安装modified版本（注意引号）
pip install "E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main (modified for 2022 WC)"

# 4. 验证
python -c "from unravel.soccer import KloppyPolarsDataset; print('成功！')"
```

**解决方案B：降级TensorFlow（不推荐）**

```bash
pip uninstall tensorflow tensorflow-intel -y
pip install tensorflow==2.14.0
set PYTHONUTF8=1
pip install E:\JerryWu\Master\SoccerAnalytics\TrackingData_literature_code\unravelsports-main
```

**缺点**：TensorFlow 2.14.0缺少新特性，可能与其他包冲突

**解决方案C：跳过unravelsports**

```bash
# 直接使用kloppy
python -c "import kloppy; print('kloppy可用！')"
```

## 环境管理

### 激活环境
```bash
cd MORPH
MORPHenv\Scripts\activate
```

### 停用环境
```bash
deactivate
```

### 删除环境
```bash
# 停用环境
deactivate

# 删除文件夹
rmdir /s MORPHenv
```

### 导出环境
```bash
pip freeze > requirements_frozen.txt
```

### 更新依赖
```bash
pip install --upgrade -r requirements.txt
```

## 下一步

环境配置完成后，请按以下顺序进行：

1. **验证环境**
   - 运行上述验证代码
   - 确保所有包正确安装

2. **测试数据加载**
   - 打开 `Step1_Contextualization_Scaling/Test/1.1_test_Convert_TrackingData.ipynb`
   - 运行第一个单元格
   - 确认数据路径正确

3. **运行测试版**
   - 完整运行测试版notebook（决赛数据）
   - 预计时间：5-10分钟
   - 验证输出文件

4. **运行通用版**
   - 运行通用版notebook（64场比赛）
   - 预计时间：1-2小时
   - 检查处理结果

## 技术支持

如遇到其他问题，请参考：
- MORPH适配指南：`MORPH_ADAPTATION_GUIDE.md`
- Kloppy文档：https://kloppy.pysport.org/
- PyTorch Geometric文档：https://pytorch-geometric.readthedocs.io/
- Pyro文档：https://pyro.ai/

---

**最后更新**: 2025-12-07  
**版本**: v1.0.0