import tempfile
import os
import subprocess
import librosa
import numpy as np
import soundfile as sf
from noisereduce import reduce_noise

def get_file_extension(filepath):
    # 使用os.path.splitext函数获取文件路径的扩展名
    _, file_extension = os.path.splitext(filepath)
    return file_extension.lower()  # 返回小写形式的文件扩展名

def convert_to_wav(input_file):
    # 创建临时文件
    # 提取原始文件的文件名
    file_name = os.path.basename(input_file)
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', prefix=file_name, delete=False) as temp_file:
        output_file = temp_file.name
        # 使用FFmpeg命令进行音频转换
        subprocess.run(['ffmpeg','-y', '-i', input_file, output_file])
        # 返回临时文件路径
        return output_file

# 原降噪函数，可以删掉
def Denoise(file):
    # 使用librosa加载音频文件
    audio_data, sample_rate = librosa.load(file, sr=None)  # sr=None表示保持原始采样率
    # 确保音频是单声道的
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # 如果音频是立体的，则取平均值以转换为单声道
    # 使用noisereduce进行降噪，这次不传入noise_clip参数
    # 这将使得noisereduce库自动估计噪音
    reduced_noise_audio = reduce_noise(y=audio_data, sr=sample_rate)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        wav_filename = temp_file.name
        sf.write(wav_filename, (reduced_noise_audio * 32767).astype(np.int16), sample_rate)
    return wav_filename
