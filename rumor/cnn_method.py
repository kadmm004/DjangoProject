# 输入词向量序列，产生一个特征图（feature map），
# 对特征图采用时间维度上的最大池化（max pooling over time）操作得到此卷积核对应的整句话的特征，
# 最后，将所有卷积核得到的特征拼接起来即为文本的定长向量表示，
# 对于文本分类问题，将其连接至softmax即构建出完整的模型。
# 在实际应用中，我们会使用多个卷积核来处理句子，
# 窗口大小相同的卷积核堆叠起来形成一个矩阵，这样可以更高效的完成运算。

# 单层
import zipfile
import os
import io
import random
import json
from _ast import arg

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Linear, Embedding
from paddle.fluid.dygraph.base import to_variable

from rumor.dict_new import SentaProcessor

train_parameters = {
    "epoch": 50,  # 训练轮次
    "batch_size": 128,  # 批次大小
    "adam": 0.006,  # 学习率
    "padding_size": 150,  # padding纬度
    "vocab_size": 4409,  # 字典大小
    "skip_steps": 100,  # 每N个批次输出一次结果
    "save_steps": 100,  # 每M个批次保存一次
    "checkpoints": "D:/study_mxd/cnn_rumor/data/"  # 保存路径
}


class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,  # 通道数
                 num_filters,  # 卷积核数量
                 filter_size,  # 卷积核大小
                 batch_size=None):  # 16
        super(SimpleConvPool, self).__init__()
        self.batch_size = batch_size
        self._conv2d = Conv2D(num_channels=num_channels,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              act='tanh')
        self._pool2d = fluid.dygraph.Pool2D(
            pool_size=(150 - filter_size[0] + 1, 1),
            pool_type='max',
            pool_stride=1
        )

    def forward(self, inputs):
        # print('SimpleConvPool_inputs数据纬度',inputs.shape) # [16, 1, 148, 128]
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        x = fluid.layers.reshape(x, shape=[self.batch_size, -1])
        return x


class CNN(fluid.dygraph.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        self.dict_dim = train_parameters["vocab_size"]
        self.emb_dim = 128  # emb纬度
        self.hid_dim = [32]  # 卷积核数量
        self.fc_hid_dim = 96  # fc参数纬度
        self.class_dim = 2  # 分类数
        self.channels = 1  # 输入通道数
        self.win_size = [[3, 128]]  # 卷积核尺寸
        self.batch_size = train_parameters["batch_size"]
        self.seq_len = train_parameters["padding_size"]
        self.embedding = Embedding(
            size=[self.dict_dim + 1, self.emb_dim],
            dtype='float32',
            is_sparse=False)
        self._simple_conv_pool_1 = SimpleConvPool(
            self.channels,
            self.hid_dim[0],
            self.win_size[0],
            batch_size=self.batch_size)
        self._fc1 = Linear(input_dim=self.hid_dim[0],
                           output_dim=self.fc_hid_dim,
                           act="tanh")
        self._fc_prediction = Linear(input_dim=self.fc_hid_dim,
                                     output_dim=self.class_dim,
                                     act="softmax")

    def forward(self, inputs, label=None):

        emb = self.embedding(inputs)  # [2400, 128]
        # print('CNN_emb',emb.shape)
        emb = fluid.layers.reshape(  # [16, 1, 150, 128]
            emb, shape=[-1, self.channels, self.seq_len, self.emb_dim])
        # print('CNN_emb',emb.shape)
        conv_3 = self._simple_conv_pool_1(emb)
        fc_1 = self._fc1(conv_3)
        prediction = self._fc_prediction(fc_1)
        if label is not None:
            acc = fluid.layers.accuracy(prediction, label=label)
            return prediction, acc
        else:
            return prediction


def draw_train_process(iters, train_loss, train_accs):
    title = "training loss/training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("loss/acc", fontsize=14)
    plt.plot(iters, train_loss, color='red', label='training loss')
    plt.plot(iters, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()


def train():
    # with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
    with fluid.dygraph.guard(place=fluid.CPUPlace()):

        processor = SentaProcessor(data_dir="D:/study_mxd/cnn_rumor/data/")

        train_data_generator = processor.data_generator(
            batch_size=train_parameters["batch_size"],
            phase='train',
            shuffle=True)

        model = CNN()
        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=train_parameters["adam"],
                                                parameter_list=model.parameters())
        steps = 0
        Iters, total_loss, total_acc = [], [], []
        for eop in range(train_parameters["epoch"]):
            for batch_id, data in enumerate(train_data_generator()):
                steps += 1
                # 转换为 variable 类型
                doc = to_variable(
                    np.array([
                        np.pad(x[0][0:train_parameters["padding_size"]],  # 对句子进行padding，全部填补为定长150
                               (0, train_parameters["padding_size"] - len(x[0][0:train_parameters["padding_size"]])),
                               'constant',
                               constant_values=(train_parameters["vocab_size"]))  # 用 <unk> 的id 进行填补
                        for x in data
                    ]).astype('int64').reshape(-1))
                # 转换为 variable 类型
                label = to_variable(
                    np.array([x[1] for x in data]).astype('int64').reshape(
                        train_parameters["batch_size"], 1))

                model.train()  # 使用训练模式
                prediction, acc = model(doc, label)
                loss = fluid.layers.cross_entropy(prediction, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                sgd_optimizer.minimize(avg_loss)
                model.clear_gradients()

                if steps % train_parameters["skip_steps"] == 0:
                    Iters.append(steps)
                    total_loss.append(avg_loss.numpy()[0])
                    total_acc.append(acc.numpy()[0])
                    print("eop: %d, step: %d, ave loss: %f, ave acc: %f" %
                          (eop, steps, avg_loss.numpy(), acc.numpy()))
                if steps % train_parameters["save_steps"] == 0:
                    save_path = train_parameters["checkpoints"] + "/" + "save_dir_" + str(steps)
                    print('save model to: ' + save_path)
                    fluid.dygraph.save_dygraph(model.state_dict(),
                                               save_path)
                # break
    draw_train_process(Iters, total_loss, total_acc)


# 开始训练
# train()

def to_eval():
    # If you want to use GPU, please try to install GPU version PaddlePaddle by: pip install paddlepaddle-gpu
    # If you only have CPU, please change CUDAPlace(0) to be CPUPlace().
    with fluid.dygraph.guard(place=fluid.CPUPlace()):
        processor = SentaProcessor(data_dir="D:/study_mxd/cnn_rumor/data/")  # 写自己的路径

        eval_data_generator = processor.data_generator(
            batch_size=train_parameters["batch_size"],
            phase='eval',
            shuffle=False)

        model_eval = CNN()  # 示例化模型
        model, _ = fluid.load_dygraph("D:/study_mxd/cnn_rumor/data/save_dir_1100.pdparams")  # 写自己的路径
        model_eval.load_dict(model)

        model_eval.eval()  # 切换为eval模式
        total_eval_cost, total_eval_acc = [], []
        for eval_batch_id, eval_data in enumerate(eval_data_generator()):
            eval_np_doc = np.array([np.pad(x[0][0:train_parameters["padding_size"]],
                                           (0, train_parameters["padding_size"] - len(
                                               x[0][0:train_parameters["padding_size"]])),
                                           'constant',
                                           constant_values=(train_parameters["vocab_size"]))
                                    for x in eval_data
                                    ]).astype('int64').reshape(-1)
            eval_label = to_variable(
                np.array([x[1] for x in eval_data]).astype(
                    'int64').reshape(train_parameters["batch_size"], 1))
            eval_doc = to_variable(eval_np_doc)
            eval_prediction, eval_acc = model_eval(eval_doc, eval_label)
            loss = fluid.layers.cross_entropy(eval_prediction, eval_label)
            avg_loss = fluid.layers.mean(loss)
            total_eval_cost.append(avg_loss.numpy()[0])
            total_eval_acc.append(eval_acc.numpy()[0])

    print("Final validation result: ave loss: %f, ave acc: %f" %
          (np.mean(total_eval_cost), np.mean(total_eval_acc)))


# 模型评估
to_eval()
