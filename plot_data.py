import matplotlib.pyplot as plt
import numpy as np
import random

path = "data/mnist/original_28x28/all_digits_all_pixels/x_train.npy"
x_train = np.load(path).reshape(-1, 28, 28)

while True:
    plt.imshow(random.choice(x_train), cmap='gray')
    plt.show()
