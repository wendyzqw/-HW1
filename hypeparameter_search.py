import os
import sys

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from util import SGD, shuffle_dataset

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

x_train = x_train[: 2000]
y_train = y_train[: 2000]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, y_train = shuffle_dataset(x_train, y_train)
x_val = x_train[: validation_num]
y_val = y_train[: validation_num]
x_train = x_train[validation_num:]
y_train = y_train[validation_num:]

def __train(lr, l2_lambda, hidden_size):
    network = MultiLayerNet(input_size=784, hidden_size_list=[hidden_size, hidden_size],
                            output_size=10, l2_lambda=l2_lambda)
    optimizer = SGD(lr=lr, exponetial_decay=True)
    max_epochs = 21
    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    test_loss_list = []

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
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            # print("epoch" + str(epoch_cnt) + ", train loss:" + str(train_loss) + ",test loss:" + str(test_loss))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
    return train_loss_list, test_loss_list

results_val = {}
results_train = {}
lr_list = [0.01, 0.05]
l2_lambda_list = [0.1, 0.05]
hidden_size_list = [128, 64]
for lr in lr_list:
    for l2_lambda in l2_lambda_list:
        for hidden_size in hidden_size_list:
            train_loss_list, val_loss_list = __train(lr, l2_lambda, hidden_size)
            print('val loss:' + str(val_loss_list[-1]) + '| lr:' + str(lr) + ',l2 lambda:' + str(l2_lambda) + ',hidden size:' + str(hidden_size))
            key = 'lr:' + str(lr) + ',l2 lambda:' + str(l2_lambda) + ',hidden size:' + str(hidden_size)
            results_val[key] = val_loss_list
            results_train[key] = train_loss_list

param_best = None
value_best = float('inf')
for key, value in results_val.items():
    value_best = min(value_best, value[-1])
    if value_best == value[-1]:
        param_best = key

print('找到的最优参数组合为：' + param_best)
