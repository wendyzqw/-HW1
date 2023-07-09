import matplotlib.pyplot as plt
from multi_layer_net import MultiLayerNet

network = MultiLayerNet(input_size=784, hidden_size_list=[64, 64],
                        output_size=10, l2_lambda=0.05)
network.load_params('params.pkl')


plt.imshow(network.params['W1'], cmap='Reds',interpolation='nearest')
plt.axis('off')
plt.show()

plt.imshow(network.params['W2'], cmap='viridis',interpolation='nearest')
plt.axis('off')
plt.show()
