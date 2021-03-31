"""!
@brief Sobel filtering simple example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

matrix = np.ndarray((11, 11), dtype=int)
matrix = 100*np.zeros((11,11))

matrix[3, 5] = 0
matrix[3, 6] = 50
matrix[4, 5] = 100
matrix[4, 6] = 150
matrix[5, 5] = 200
matrix[5, 6] = 250

img = matrix.astype(np.uint8)
s = 3  # set kernel size

kernel = np.ones((s, s), np.float32)/(s**2)
img2 = cv2.filter2D(img, -1, kernel=kernel)
plt.subplot(2, 1, 1); plt.imshow(img, cmap='gray'); plt.title("original")
plt.subplot(2, 1, 2); plt.imshow(img2, cmap='gray'); plt.title("averaging")
plt.show()
