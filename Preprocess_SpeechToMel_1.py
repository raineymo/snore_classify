import os
import numpy as np
from pathlib import Path
import librosa

def wav_to_mel(wavpath):# 计算log FBank特征
     # 计算log FBank特征
    y, sr = librosa.load(wavpath, sr=None)  # sr=None表示保持原始采样率
    amp =  librosa.stft(y,n_fft=1024,hop_length=256, win_length=1024)
    # 转换为对数刻度
    amp = librosa.power_to_db(amp, ref=np.max) ## -8 到1 之间
    return amp

def extract_amptitude(wav_datadir,output_dir):
    src_wavp = Path(wav_datadir)  # Path
    # 提取特征
    wavpaths = [x for x in src_wavp.rglob('*.wav') if x.is_file()]
    ttsum = len(wavpaths) # 总语音数量
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp)
        relative_path = os.path.relpath(the_wavpath, wav_datadir)
        filename = os.path.basename(the_wavpath)
        # 根据相对路径确定输出文件夹
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)
        the_melpath = os.path.join(output_subdir, filename.replace('.wav', '.npy'))
        # 计算log FBank特征
        y, sr = librosa.load(the_wavpath, sr=32000)  # sr=None表示保持原始采样率
        amp =  librosa.stft(y,n_fft=1024,hop_length=256, win_length=1024)
        # 转换为对数刻度
        amp = librosa.power_to_db(amp, ref=np.max)
        print(f"amp shape:{amp.shape},{k}|{ttsum}")
        np.save(the_melpath,amp)
        print("提取完毕  "+the_melpath)

if __name__ == '__main__':
    # 提取 mel
    # wav 文件的 数据集的目录
    src = './dataset'
    # melspec 文件夹
    tar = './e1/feat1'
    extract_amptitude(src,tar)