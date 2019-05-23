import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Directions matrix for calculating new bottom-left
# [[['k-1' related], ['k' related]], ...]
D = np.array([
    [[0, 0], [-1, 0]], # left
    [[0, 0], [ 0, -1]], # down
    [[1, 0], [ 0, 0]], # right
    [[0, 1], [ 0, 0]]  # up
])

# Calculate Fibonacci sequence
n = 8
M = [None] * (n + 1)
M[0], M[1] = 0, 1
for k in range(2, n + 1):
    M[k] = M[k-1] + M[k-2]

# figure, axes
fig = plt.figure()
axe = fig.add_subplot(111, aspect='equal')

# 1st rect
k = 1
bl = bl_prev = (0, 0)
axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False))

# Other rects
for k in range(2, n + 1):
    bl = (
        bl_prev +
        D[k%4][0] * [M[k-1], M[k-1]] +
        D[k%4][1] * [M[k], M[k]]
    )
    bl_prev = np.min([bl_prev, bl], axis=0)

    axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False))

plt.xlim(-M[k] - M[k-1], M[k])
plt.ylim(-M[k-1], M[k-1] + M[k-2])
plt.show()

