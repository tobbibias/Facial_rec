import os
import cv2
from matplotlib import pyplot
path = "img1.png"

image = cv2.imread(path)
image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

pyplot.imshow(image2)
pyplot.savefig("test.png")


