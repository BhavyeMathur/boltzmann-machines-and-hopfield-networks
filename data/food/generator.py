import cv2
import numpy as np

def read_image(path):
    im = cv2.imread(path)[:, :, 1]
    im = im.astype(bool).astype("uint16")
    return im.flatten()

donut = read_image("donut.png")
burger = read_image("burger.png")
fries = read_image("fries.png")
popcorn = read_image("popcorn.png")
sandwich = read_image("sandwich.png")

data = np.stack((burger, donut, popcorn), axis=0)
np.save("food.npy", data)
print(data.shape)
