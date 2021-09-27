# 导入必要的包
import os
from multiprocessing import cpu_count

import numpy
import numpy as np
import shutil
import random
import paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt


# 首先生成数据字典
def create_dict(data_path, dict_path):
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate()

    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print("数据字典生成完成！", '\t', '字典长度为：', len(dict_list))


# 创建序列化表示的数据,并按照一定比例划分训练数据与验证数据
def create_data_list(data_list_path):
    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, \
            open(os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            title = line.split('\t')[-1].replace('\n', '')
            lab = line.split('\t')[0]
            t_ids = ""
            if i % 8 == 0:
                for s in title:  # 这里有参数不一样
                    temp = str(dict_txt[s])
                    t_ids = t_ids + temp + ','
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_eval.write(t_ids)
            else:
                for s in title:
                    temp = str(dict_txt[s])
                    t_ids = t_ids + temp + ','
                t_ids = t_ids[:-1] + '\t' + lab + '\n'
                f_train.write(t_ids)
            i += 1

    print("数据列表生成完成！")


data_root_path = "D:/study_mxd/cnn_rumor/data/"  # 写自己的路径
data_path = os.path.join(data_root_path, 'all_data.txt')
dict_path = os.path.join(data_root_path, "dict.txt")
# 创建数据字典
create_dict(data_path, dict_path)

# 先清空已有txt文件
with open(os.path.join(data_root_path, 'train_list.txt'), 'w', encoding='utf-8') as f_eval:
    f_eval.seek(0)
    f_eval.truncate()

with open(os.path.join(data_root_path, 'eval_list.txt'), 'w', encoding='utf-8') as f_train:
    f_train.seek(0)
    f_train.truncate()

# 创建数据列表
create_data_list(data_root_path)


# 定义数据读取器
def data_reader(file_path, phrase, shuffle=False):
    all_data = []
    with open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) != 2:
                continue
            label = int(cols[1])

            wids = cols[0].split(",")
            all_data.append((wids, label))

    if shuffle:
        if phrase == "train":
            random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label

    return reader


class SentaProcessor(object):
    def __init__(self, data_dir, ):
        self.data_dir = data_dir

    def get_train_data(self, data_dir, shuffle):
        return data_reader((self.data_dir + "train_list.txt"),
                           "train", shuffle)

    def get_eval_data(self, data_dir, shuffle):
        return data_reader((self.data_dir + "eval_list.txt"),
                           "eval", shuffle)

    def data_generator(self, batch_size, phase='train', shuffle=True):
        if phase == "train":
            return paddle.batch(
                self.get_train_data(self.data_dir, shuffle),
                batch_size,
                drop_last=True)
        elif phase == "eval":
            return paddle.batch(
                self.get_eval_data(self.data_dir, shuffle),
                batch_size,
                drop_last=True)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'eval']")
