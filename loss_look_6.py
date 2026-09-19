import pickle
from matplotlib import pyplot as plt
from pathlib import Path


def plot_loss_bylogtxt(txt_path, ver):
    with open(txt_path, encoding='utf-8') as f2:
        lines = list(f2.readlines())
    loss_lines = lines[:]  # 将这一行下面所有行保存下来。用于作图。

    # 1.2 寻找 数据项 的key
    loss_keys = []
    for line in loss_lines:
        ## 忽略掉 “当前进程的内存使用”数据项
        ## k_v.split(':')[0] 是从 A:B中取出A.
        l_keys = [k_v.split(':')[0] for k_v in line.split(',')[:-1]]
        loss_keys += l_keys
    loss_keys = list(set(loss_keys))  ## 取集合得到字典的键。
    # 创建字典
    loss_dict = {}
    for k in loss_keys:
        loss_dict[k] = []

    # 1.3 接下来，将每一个键对应的一组数值，加入到列表中
    for line in loss_lines:
        l_keys_values = [(k_v.split(':')[0], k_v.split(':')[1]) for k_v in line.split(',')[:-1]]
        # 将每个 A:B,添加到字典。
        for a_b in l_keys_values:
            loss_dict[a_b[0]].append(float(a_b[1]))  # 字符转浮点数
    ## 则loss dict 保存了全部数据项的values和keys。
    '''
    loss dict:
    {
    'epoch':[0,0,0,0,0,...]
    'step':[0,1,2,3....]
    'loss1':[10,8,7,6,....]
    'loss2':[100,90,....]
    }
    '''

    # 创建保存图片的文件夹
    save_file_dir = Path('Lossfigure/Loss_Curves_Figures_{}'.format(ver))
    save_file_dir.mkdir(parents=True, exist_ok=True)

    # 作图
    for k, v in loss_dict.items():
        plt.figure()
        plt.title(str(k))
        plt.plot(range(len(v)), v)
        plt.savefig(str(save_file_dir / f"{k}.png"))
        plt.show()

if __name__ == '__main__':
    ver = "e1"
    pa = f'e1/train_log.txt'
    plot_loss_bylogtxt(pa, ver)
    pass
