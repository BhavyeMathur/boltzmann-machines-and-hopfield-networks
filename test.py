import numpy as np
import torch
import matplotlib.pyplot as plt

tensor = torch.load("data/mnist/training.pt")[0]
tensor = tensor.reshape(-1, 28 * 28)
data = (tensor.numpy() > 0).astype("uint16")
np.save("data/mnist/mnist.npy", data)
print(tensor.shape)

data = np.load("data/mnist/mnist.npy")
print(data.shape)

sample = data[100].reshape(28, 28)
plt.imshow(sample)
plt.show()
