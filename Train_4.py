import numpy as np
import random
import torch.nn as nn
from torch.utils import data
from pathlib import Path
import torch
from Model_3 import SpeakerEncoder
from Mydataset_2 import MeldataSet
from Mydataset_2 import segment2d
import torch.nn.functional as F
from Mydataset_2 import max_minnorm

# 打印模型的信息
def print_network(model, name):
    num_params = sum(p.numel() for p in model.parameters())
    print("Model {},the number of parameters: {}".format(name, num_params))

#  打印日志
def write_line2log(log_dict: dict, filedir, is_write=True, isprint=True):
    strp = ''
    for key, value in log_dict.items():
        witem = '{}'.format(key) + ':{},'.format(value)
        strp += witem
    strp += "\n"
    if is_write:
        with open(filedir, 'a', encoding='utf-8') as f:
            f.write(strp)
    if isprint:
        print(strp)
    pass

# 随机种子
def same_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 定义一个函数用于训练声纹识别模型
def train(hps):
    # 创建实验文件夹、模型检查点文件夹以及训练和测试日志文件路径
    setup_experiment_files(hps)
    # 初始化数据集和模型
    train_dataset, num_classes, train_mel_dataset = initialize_dataset_and_model(hps)
    # 准备训练所需的设备和参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 训练设备
    batch_size = hps["batch_size"]
    model_save_interval = hps["model_save_inter"]
    train_loader = data.DataLoader(train_mel_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # 定义模型损失函数和优化器
    model, loss_function, optimizer = initialize_model(num_classes, device)
    # 开始训练过程
    start_training(model, loss_function, optimizer, train_loader, device, model_save_interval, train_mel_dataset, hps)

# 设置实验所需的文件路径和目录结构
def setup_experiment_files(hps):
    experiment_dir = Path(hps["experiment_name"])
    checkpoints_dir = experiment_dir / "model_checkpoints"
    train_log_path = experiment_dir / "train_log.txt"
    test_log_path = experiment_dir / "test_log.txt"
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    train_log_path.touch()
    test_log_path.touch()

# 初始化数据集和模型
def initialize_dataset_and_model(hps):
    train_mel_dataset = MeldataSet(datadir=hps["cls_datadir"],
                                   melspec_len=hps["spec_seglen"],
                                   test_ratio=0.05,
                                   hps=hps
                                   )
    num_classes = train_mel_dataset.num_class
    return train_mel_dataset, num_classes, train_mel_dataset

# 初始化模型、损失函数和优化器
def initialize_model(num_classes, device):
    model_params_config = {
        "SpeakerEncoder": {
            "c_in": 513,
            "c_h": 128,
            "c_out": 128,
            "kernel_size": 5,
            "bank_size": 8,
            "bank_scale": 1,
            "c_bank": 128,
            "n_conv_blocks": 6,
            "n_dense_blocks": 6,
            "subsample": [1, 2, 1, 2, 1, 2],
            "act": 'relu',
            "dropout_rate": 0,
            "num_class": num_classes
        }
    }
    model = SpeakerEncoder(**model_params_config['SpeakerEncoder']).to(device)
    print_network(model, 'cls')
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, betas=(0.99, 0.999))
    return model, loss_function, optimizer

# 开始训练过程
def start_training(model, loss_function, optimizer, train_loader, device, model_save_interval, train_mel_dataset, hps):
    epoch_num = 0
    train_iter = 0
    print("开始训练")
    for epoch_num in range(hps["total_epoch_num"]):
        for batch in train_loader:
            model.train()
            train_iter += 1
            # 将数据加载到设备上
            mels, labels = batch
            mels = mels.to(device)
            labels = labels.to(device)
            # 进行前向传播、计算损失
            pred_prob = model(mels)
            batch_loss = loss_function(pred_prob, labels)
            ## 更新神经网络的参数
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # 记录训练损失和准确率
            record_training_progress(epoch_num, train_iter, batch_loss, labels, pred_prob, hps)
            # 定期评估模型性能并记录日志
            if train_iter % hps["eval_interval"] == 0:
                evaluate_and_log(model, train_mel_dataset, device, hps)
        # 保存模型参数
        if train_iter % model_save_interval == 0:
            save_model(model, epoch_num, hps)

# 记录训练过程中的损失和准确率
def record_training_progress(epoch_num, train_iter, batch_loss, labels, pred_prob, hps):
    ep_filedir = Path(hps["experiment_name"])
    log_trainpath = ep_filedir / "train_log.txt"
    pred_index = torch.max(pred_prob, dim=1)[1]
    batch_acc = (pred_index == labels).sum() / hps["batch_size"]
    loss_curves = {
        "epoch": epoch_num,
        "steps": train_iter,
        "loss": batch_loss.item(),
        "batch_acc": batch_acc
    }
    write_line2log(loss_curves, log_trainpath, is_write=True, isprint=True)

# 评估模型性能并记录日志
def evaluate_and_log(model, train_mel_dataset, device, hps):
    ep_filedir = Path(hps["experiment_name"])
    log_testpath = ep_filedir / "test_log.txt"
    eval_acc = compute_accuracy(model, train_mel_dataset.eval_samples, device, hps["spec_seglen"], hps["is_norm"],
                                train_mel_dataset)
    test_acc = compute_accuracy(model, train_mel_dataset.test_samples, device, hps["spec_seglen"], hps["is_norm"],
                                train_mel_dataset)
    write_evaluation_results_to_log(eval_acc, test_acc, log_testpath)

# 计算模型的准确率
def compute_accuracy(model, data_loader, device, spec_seglen, is_norm, train_mel_dataset):
    model.eval()
    right_predictions = 0
    total_samples = 0
    for datasa in data_loader:
        data, label = datasa
        mel = segment2d(np.load(data), seglen=spec_seglen)
        if is_norm:
            mel = torch.FloatTensor(max_minnorm(mel)).to(device)
        label = torch.LongTensor([label]).to(device)
        mel = mel.unsqueeze(0)
        output_prob = model(mel)
        _, predicted_labels = torch.max(F.softmax(output_prob, dim=-1), dim=-1)
        if predicted_labels == label:
            right_predictions += 1
        total_samples += 1
    test_acc = right_predictions / train_mel_dataset.test_nums
    return test_acc

# 将评估结果记录到日志中
def write_evaluation_results_to_log(eval_acc, test_acc, log_testpath):
    evaluation_results = {
        "eval_acc": eval_acc,
        "test_acc": test_acc
    }
    write_line2log(evaluation_results, log_testpath, is_write=True, isprint=True)
    print("-" * 30)

# 保存模型参数
def save_model(model, epoch_num, hps):
    ep_filedir = Path(hps["experiment_name"])
    ckpts_dir = ep_filedir / "model_checkpoints"
    modelsavep = str(ckpts_dir / "epoch_{:06}.pt".format(epoch_num))
    torch.save({"model": model.state_dict()}, modelsavep)

if __name__ == '__main__':
    same_seeds(2000)
    # 神经网络训练。
    hps = {
        # 文件夹相关
        "experiment_name": "e1",  # 实验结果文件夹
        # 数据相关
        "cls_datadir": "./e1/feat1",  # 频谱文件夹
        "spec_seglen": 800,
        "sample_rate": 16000,
        # 训练相关:
        "batch_size": 32,
        "total_epoch_num": 50,  ## 跑的轮数
        "eval_interval": 500,  # 每隔多少步验证一次，验证准确率和测试准确率。
        "model_save_inter": 10,  # 模型隔多少步存储一次。
        "is_norm": True,

    }
    train(hps=hps)
    pass
