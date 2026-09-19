import torch
from Model_3 import SpeakerEncoder
from Mydataset_2 import max_minnorm, segment2d
from Preprocess_SpeechToMel_1 import wav_to_mel

def test_acc(evaldatadir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 训练设备
    # 模型文件的路径
    model_path = "./e1/model_checkpoints/epoch.pt"
    # 加载分类表
    spk_indexlist = ['f', 't']
    # 加载神经网络模型
    model_params_config = {"SpeakerEncoder":
                               {"c_in": 513,
                                "c_h": 128,
                                "c_out": 128,
                                "kernel_size": 5,
                                "bank_size": 8,
                                "bank_scale": 1,
                                "c_bank": 128,
                                "n_conv_blocks": 6,
                                "n_dense_blocks": 6,
                                "subsample": [1, 2, 1, 2, 1, 2],  # 下采样的主要功能：缩小时间帧
                                "act": 'relu',
                                "dropout_rate": 0,
                                "num_class": 2}
                           }
    model = SpeakerEncoder(**model_params_config['SpeakerEncoder'])  # 初始化一个模型
    model.load_state_dict(torch.load(model_path)["model"])  # 加载模型
    model = model.to(device)
    mel = wav_to_mel(evaldatadir)
    mel = torch.FloatTensor(max_minnorm(segment2d(mel, seglen=800))).unsqueeze(0).to(device)
    pred_prob = model(mel)
    pred_index = torch.max(pred_prob, dim=1)[1].item()  # 求最大概率的下标
    # 下标转化为类别
    pred_class = spk_indexlist[pred_index]  ## 模型预测
    # 真实类别：文件名的最后一个字母
    s = f"测试文件:{str(evaldatadir)},模型预测类别：{pred_class}"
    print(s)
    return pred_class

def test():
    # 测试语音文件夹 路径
    eval_datadir = "test/1-20545-A.wav"
    test_acc(evaldatadir=eval_datadir,)

if __name__ == '__main__':
    # 运行test对 test 文件夹下的语音进行测试
    test()


