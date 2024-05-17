# 导入库
import os
from random import seed  # 用于固定每次生成的随机数都是确定的（伪随机数）
from random import randrange  # 用于生成随机数
import shutil

def train_val_split(dataset, train=0.8):
    """
    该函数用于划分训练集和测试集
    Parameters
    ----------
    dataset : 二维列表
        传入需要划分成训练集和测试集的数据集.
    train : 浮点数
        传入训练集占整个数据集的比例.默认是0.8.
    Returns
    -------
    train_basket : 二维列表
        划分好的训练集.
    dataset_copy : 二维列表
        划分好的测试集.
    """
    # 创建一个空列表用于存放后面划分好的训练集
    train_basket = list()
    # 根据输入的训练集的比例计算出训练集的大小（样本数量）
    train_size = train * len(dataset)
    # 复制出一个新的数据集来做切分，从而不改变原始的数据集
    dataset_copy = list(dataset)
    # 执行循环判断，如果训练集的大小小于所占的比例，就一直往训练集里添加数据
    while len(train_basket) < train_size:
        # 通过randrange()函数随机产生训练集的索引
        random_choose = randrange(len(dataset_copy))
        # 根据上面生成的训练集的索引将数据集中的样本加到train_basket中
        # 注意pop函数会根据索引将数据集中的样本给移除，所以循环结束之后剩下的样本就是测试集
        train_basket.append(dataset_copy.pop(random_choose))
    return train_basket, dataset_copy

def remove_file(old_path_list, new_path):
    for file in old_path_list:
        source_file = file
        file_name = file.split('/')[-1]
        target_file = os.path.join(new_path, file_name)
        print('source_file:', source_file)
        print('target_file:', target_file)
        shutil.move(source_file, target_file)


# 主函数
if '__main__' == __name__:
    # 定义一个随机种子，使得每次生成的随机数都是确定的（伪随机数）
    seed(666)
    image_dir = r'/media/Data/yanxc/Liver_vessel/pre_data/'
    dataset = [os.path.join(image_dir, x)
                   for x in os.listdir(image_dir)]
    # print(len(dataset))
    # 调用手动编写的train_test_split函数划分训练集和测试集
    train, val = train_val_split(dataset, train=0.8)
    # print(train,len(train))
    remove_file(train,'/media/Data/yanxc/Liver_vessel/pre_data/train/',)
    remove_file(val, '/media/Data/yanxc/Liver_vessel/pre_data/val/', )
