import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import create_unique_color_uchar

# Draw
fig = plt.figure()
axe = fig.add_subplot(111, aspect='equal')

# Directions matrix for calculating bottom-left of `k` based on `k-1`'s
# [[['k-1' related], ['k' related]], ...]
D = np.array([
    [[0, 0], [-1, 0]],  # left
    [[0, 0], [ 0, -1]], # down
    [[1, 0], [ 0, 0]],  # right
    [[0, 1], [ 0, 0]]   # up
])

# Directions matrix for calculating centroid of `k` based on bottom-left
# centroid is needed for drawing 1/4 of the circle for k-th square
D_circle = np.array([
    [1, 0], # left
    [1, 1], # down
    [0, 1], # right
    [0, 0], # up
])

# Init Fibonacci sequence
n = 8
M = [None] * (n + 1)
M[0], M[1] = 0, 1

# 1st square
k = 1
bl = bl_prev = (0, 0)
color = np.array(create_unique_color_uchar(k)) / 255
axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False, color=color))


# Other squares
for k in range(2, n + 1):
    M[k] = M[k-1] + M[k-2]

    bl = (
        bl_prev +
        D[k%4][0] * [M[k-1], M[k-1]] +
        D[k%4][1] * [M[k], M[k]]
    )
    centroid = bl + D_circle[k%4] * [M[k], M[k]]

    bl_prev = np.min([bl_prev, bl], axis=0)

    color = np.array(create_unique_color_uchar(k)) / 255
    axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False, color=color))

# Draw
plt.xlim(-M[k] - M[k-1], M[k])
plt.ylim(-M[k-1], M[k-1] + M[k-2])
plt.show()

