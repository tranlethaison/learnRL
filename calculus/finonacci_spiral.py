import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

from utils import create_unique_color_uchar


# Directions matrix for calculating new top-left
# [[['k-1' related], ['k' related]], ...]
D = np.array([
    [[0, 0], [-1, 0]], # left
    [[0, 1], [ 0, 0]], # down
    [[1, 0], [ 0, 0]], # right
    [[0, 0], [ 0,-1]]  # up
])

# Calculate Fibonacci sequence
n = 12
M = [None] * (n + 1)
M[0], M[1] = 0, 1
for k in range(2, n + 1):
    M[k] = M[k-1] + M[k-2]

# Background
size = np.array((M[n] + M[n-1], M[n] + M[n-1]))
start = size // 2
bg = np.full((size[1],size[0],3), 255, np.uint8)

# Style
fontFace = cv.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 1
thickness = 1

# 1st rect
k = 1
tl = start
tl_prev = tl
br = tl + M[k]

color = create_unique_color_uchar(k)
cv.rectangle(bg, tuple(tl), tuple(br), color, thickness)
#cv.putText(bg, str(k), (tl[0],br[1]), fontFace, fontScale, color)

# Other rects
for k in range(2, n + 1):
    tl = (
        tl_prev +
        [M[k-1], M[k-1]] * D[k%4][0] +
        [M[k], M[k]] * D[k%4][1]
    )
    tl_prev = tl_prev if sum(tl_prev) < sum(tl) else tl
    br = tl + M[k]

    color = create_unique_color_uchar(k)
    cv.rectangle(bg, tuple(tl), tuple(br), color, thickness)
    #cv.putText(bg, str(k), (tl[0],br[1]), fontFace, fontScale, color)


plt.figure()
plt.imshow(bg)
plt.axis(False)
plt.show()

