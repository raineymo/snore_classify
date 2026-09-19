# snore_classify · 鼾声与噪音分类系统

基于深度学习的鼾声 / 噪音二分类系统。给定一段音频，模型判断其是否为鼾声。
项目同时提供：命令行推理脚本、PySide2 图形界面（支持上传 / 录音 / 降噪 / 波形可视化），以及训练与可视化工具。

## 功能特性

- 鼾声判定：将音频 log 功率谱图输入 CNN，输出「鼾声 / 非鼾声」二分类结果。
- 特征提取：从原始 `.wav` 自动计算 log 功率谱图并保存为 `.npy`，训练推理共用。
- 模型训练：自带数据划分（训练 / 验证 / 测试）、训练日志与模型检查点保存。
- 图形界面：上传音频（wav/mp3/m4a/flac/ogg/flv）或实时录音 → 自动转 wav + 降噪 → 播放 / 波形显示 → 判定。
- 训练可视化：从日志一键绘制 loss / 准确率曲线。

## 项目结构

snore_classify-master/
├── Preprocess_SpeechToMel_1.py   # 特征提取：wav → log 功率谱图 .npy
├── Mydataset_2.py                # 数据集类（分段、归一化、划分）
├── Model_3.py                    # 模型定义 SpeakerEncoder（CNN）
├── Train_4.py                    # 训练脚本
├── test_5.py                     # 命令行推理 / 测试
├── app_gui.py                    # PySide2 图形界面
├── convert_to_wav_forui.py       # 格式转换 + 降噪辅助
├── loss_look_6.py                # 训练曲线可视化
├── ui/stats.ui                  # 界面布局文件
├── css/Qui.css                   # 界面样式
├── dataset/                     # 原始音频（f=非鼾声, t=鼾声）
│   ├── f/
│   └── t/
├── e1/                          # 实验目录（特征、日志、模型权重）
│   ├── feat1/                   # 提取出的 .npy 谱图（f/, t/）
│   ├── model_checkpoints/       # 模型权重 epoch.pt（见下方说明）
│   ├── train_log.txt            # 训练日志
│   └── test_log.txt             # 评估日志
├── audio/                       # 录音输出 test.wav
└── test/                        # 测试音频

## 环境依赖

- Python 3.8+
- PyTorch
- librosa, numpy, soundfile
- noisereduce（推理 / GUI 降噪）
- ffmpeg（系统安装，用于非 wav 格式转换）
- PySide2, pyaudio, pygame, matplotlib, qt_material（图形界面）

安装示例（pip）：

    pip install torch librosa numpy soundfile noisereduce PySide2 pyaudio pygame matplotlib qt_material

并确保 `ffmpeg` 已在 PATH 中。

## 数据准备

将音频按类别放入 `dataset/` 下的两个子目录：

- `dataset/t/` —— 鼾声样本（t = 鼾声）
- `dataset/f/` —— 非鼾声 / 噪音样本（f = 非鼾声）

子目录名称 `t` / `f` 即模型输出的类别标签。

## 使用流程

### 1. 特征提取

将 `dataset/` 下所有 `.wav` 转为 log 功率谱图并保存到 `e1/feat1/`（保持 f/t 目录结构）：

    python Preprocess_SpeechToMel_1.py

特征参数：STFT `n_fft=1024, hop_length=256, win_length=1024`，再做 `power_to_db`。

### 2. 训练

    python Train_4.py

- 自动划分训练 / 验证 / 测试集（测试占比 `test_ratio=0.05`）。
- 每条谱图固定截取 `spec_seglen=800` 帧，做 min-max 归一化。
- 每 `model_save_inter=10` 步保存检查点到 `e1/model_checkpoints/epoch_XXXXXX.pt`。
- 训练 / 评估指标写入 `e1/train_log.txt`、`e1/test_log.txt`。

主要超参：`batch_size=32`，`total_epoch_num=50`，优化器 `Adam(lr=1e-4, betas=(0.99,0.999))`，损失 `CrossEntropyLoss`，随机种子 `2000`。

### 3. 命令行推理 / 测试

    python test_5.py

默认对 `test/1-20545-A.wav` 进行判定，输出「是 / 否（鼾声）」。
推理自己的音频：修改 `test_5.py` 中 `test()` 的 `eval_datadir`，或调用 `test_acc("你的音频.wav")`。

### 4. 图形界面

    python app_gui.py

支持：上传音频（多种格式自动转 wav + 降噪）、实时录音（64kHz）、波形显示、播放 / 暂停 / 音量调节，以及鼾声判定（是 / 否）。

### 5. 训练曲线可视化

    python loss_look_6.py

读取 `e1/train_log.txt`，绘制并保存 loss / 准确率等曲线到 `Lossfigure/`。

## 模型说明

模型 `SpeakerEncoder`（`Model_3.py`）为卷积结构：卷积 bank（多尺度卷积拼接）→ 卷积块（残差 + 下采样）→ 全局平均池化 → 全连接块 → 分类层。

## 关于模型权重 `e1/model_checkpoints/epoch.pt`

训练好的权重文件较大，仓库中未直接提交，而是拆分为两个分片：

- `e1/model_checkpoints/epoch.pt.aa`
- `e1/model_checkpoints/epoch.pt.ab`

使用前请先拼接还原：

    cd e1/model_checkpoints
    cat epoch.pt.aa epoch.pt.ab > epoch.pt

Windows（PowerShell）：

    $ba = [System.IO.File]::ReadAllBytes("epoch.pt.aa")
    $bb = [System.IO.File]::ReadAllBytes("epoch.pt.ab")
    [System.IO.File]::WriteAllBytes("epoch.pt", ($ba + $bb))
