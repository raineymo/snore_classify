import torch
import numpy as np
from pathlib import Path
from torch.utils import data

# 均值方差归一化。 帮助训练效果比较大的提升
def max_minnorm(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val != min_val:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = np.zeros_like(x)
    return x

# 将输入的mel频谱特征矩阵进行分段和填充，使其达到固定的长度seglen。
def segment2d(x, seglen=128):
    L = x.shape[1]
    if L < seglen:
        pad_len = seglen - L
        y = np.pad(x, ((0, 0), (0, pad_len)), mode='constant')
    elif L == seglen:
        y = x
    else:
        start = np.random.randint(L - seglen + 1)
        y = x[:, start:start + seglen]
    return y

class MeldataSet(data.Dataset):
    def __init__(self, datadir, melspec_len, test_ratio, hps):
        self.datadir = Path(datadir)  # 将一个目录的字符串，变成Path类的对象
        self.melspec_len = melspec_len
        self.test_ratio = test_ratio  # 测试集占总集比例
        self.hps = hps
        # 1.提取 分类表。
        self.spk_index_list = [subdir.name for subdir in self.datadir.glob("*") if subdir.is_dir()]
        self.num_class = len(self.spk_index_list)  # 分类的数量
        # 语音划分成验证集、训练集、测试集。
        self.train_samples = []
        self.eval_samples = []
        self.test_samples = []
        for subspk_dir in [subdir for subdir in self.datadir.glob("*") if subdir.is_dir()]:
            # 读单分类的语音路径。进行划分。
            self.spk_samples = [[p, self.spk_index_list.index(p.parts[-2])] for p in subspk_dir.rglob("*.npy") if
                                p.is_file()]
            test_num = int(self.test_ratio * len(self.spk_samples))
            self.train_samples += self.spk_samples[:-test_num]
            self.eval_samples += self.spk_samples[-test_num * 2:-test_num]
            self.test_samples += self.spk_samples[-test_num:]
        self.train_nums = len(self.train_samples)
        self.eval_nums = len(self.eval_samples)  # evaluation
        self.test_nums = len(self.test_samples)
        # 保存 spkindex list
        # 存储参数文件本身
        index_listpath = Path(self.hps["experiment_name"]) / "spk_index_list.txt"
        index_listpath.touch()
        f = index_listpath.open('w')
        for i in range(len(self.spk_index_list)):
            f.write(f"{self.spk_index_list[i]}|{i}\n")
        print("训练分类表:", self.spk_index_list)

    def __getitem__(self, idx):  # 样本数据的索引：样本 ：（频谱，标签）
        melp, label = self.train_samples[idx]  # 语音的路径
        # 读取频谱
        mel = np.load(melp)  ## 计算得梅尔频谱 np.array # [513, len]
        mel = segment2d(mel, seglen=self.melspec_len)  # 截断或者补零，每条语音固定为 self.melspec_len
        # 归一化
        if self.hps["is_norm"]:
            mel = max_minnorm(mel)
        label = torch.LongTensor([label])
        return mel, label.squeeze_()

    def __len__(self):
        return self.train_nums

if __name__ == '__main__':
    hps = {
        # 文件夹相关
        "experiment_name": "e1",  # 实验结果放在哪个文件夹下面？(需要你自己修改)
        # 数据相关
        "cls_datadir": "./e1/feat1",  # 频谱文件夹。(需要你自己修改)
        "spec_seglen": 800,
        "sample_rate": 16000,
        # 训练相关:
        "batch_size": 32,
        "total_epoch_num": 50,  # 跑的轮数
        "eval_interval": 500,  # 每隔多少步验证一次，验证准确率和测试准确率。
        "model_save_inter": 10,  # 模型隔多少步存储一次。
        "is_norm": True,
    }

    train_mel_dataset = MeldataSet(datadir="./e1/feat1",
                                   melspec_len=200,
                                   test_ratio=0.05,
                                   hps=hps
                                   )

    for (mel, label) in train_mel_dataset:
        print(mel.shape, label)
    pass
