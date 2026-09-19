# 模型权重 reassembly / 模型权重还原

由于网络对单次上传体积的限制，`epoch.pt`（约 14.4 MB）被拆分为两个二进制分片提交到仓库：

- `epoch.pt.aa`
- `epoch.pt.ab`

克隆/下载本目录后，按字节顺序拼接两个分片即可还原为完整的 `epoch.pt`（与原始文件逐字节一致）：

```bash
# Linux / macOS / Git Bash
cat epoch.pt.aa epoch.pt.ab > epoch.pt
```

```powershell
# Windows PowerShell (二进制拼接)
$ba = [System.IO.File]::ReadAllBytes("epoch.pt.aa")
$bb = [System.IO.File]::ReadAllBytes("epoch.pt.ab")
[System.IO.File]::WriteAllBytes("epoch.pt", ($ba + $bb))
```

还原后即可被 `Model_3.py` / `Train_4.py` 等正常加载。`.gitignore` 已忽略 `epoch.pt`，避免重复提交完整大文件。
