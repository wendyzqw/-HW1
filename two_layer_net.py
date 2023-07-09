import os
import sys

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from util import SGD

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

network = MultiLayerNet(input_size=784, hidden_size_list=[64, 64],
                        output_size=10, l2_lambda=0.05)
optimizer = SGD(lr=0.05, exponetial_decay=True)
max_epochs = 21
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
test_loss_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)
    optimizer.update(network.params, grads, epoch_cnt)

    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_train, y_train)
        test_loss = network.loss(x_test, y_test)
        test_acc = network.accuracy(x_test, y_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("epoch" + str(epoch_cnt) + ", train loss:" + str(train_loss) + ",test loss:" + str(test_loss))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

network.save_params('params.pkl')
print('模型参数已保存！')

def loss_curve():
    x = np.arange(max_epochs)
    plt.plot(x, train_loss_list, marker='o', label='train',markevery=10)
    plt.plot(x, test_loss_list, marker='s', label='test', markevery=10)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(train_loss_list[0], test_loss_list[0]) + 1)
    plt.legend(loc = 'lower right')
    plt.show()

def acc_curve():
    x = np.arange(max_epochs)
    plt.plot(x, test_acc_list)
    plt.xlabel('epochs')
    plt.ylabel('test_accuracy')
    plt.ylim(0, 1.0)
    plt.show()

loss_curve()
acc_curve()
