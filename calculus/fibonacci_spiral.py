import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import create_unique_color_uchar

# Plotting prepare
fig = plt.figure()
axe = fig.add_subplot(111)

# Directions matrix for calculating bottom-left of k-th square
# wrt [[M[k-1], M[k-1]], [M[k], M[k]]]
D = np.array([
    [[0, 0], [-1,  0]],  # left
    [[0, 0], [ 0, -1]],  # down
    [[1, 0], [ 0,  0]],  # right
    [[0, 1], [ 0,  0]]   # up
])

# 1/4 of the circle for k-th square
f = lambda x, r: (r**2 - x**2)**0.5
# Directions matrix for calculating centroid
D_centroid = np.array([
    [1, 0], # left
    [1, 1], # down
    [0, 1], # right
    [0, 0], # up
])
# Directions matrix of X range wrt [r, r]
# ignore point [0, 0] for the sake of simple
D_X = np.array([
    [-1, 0], # left  => top-left
    [-1, 0], # down  => bottom-left
    [ 0, 1], # right => bottom-right
    [ 0, 1], # up    => top-right
])
# Directions matrix of Y range wrt to Y range
D_Y = np.array([
     1, # left  => top-left
    -1, # down  => bottom-left
    -1, # right => bottom-right
     1, # up    => top-right
])

# Init Fibonacci sequence
n = 44
M = [None] * (n + 1)
M[0], M[1] = 0, 1

# 1st Fibonacci
k = 1
bl = bl_prev = (0, 0)
color = np.array(create_unique_color_uchar(k)) / 255
axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False, color=color))

# k-th Fibonacci
for k in range(2, n + 1):
    M[k] = M[k-1] + M[k-2]
    direction = k % 4

    # square's bottom-left
    bl = (
        bl_prev +
        D[direction][0] * [M[k-1], M[k-1]] +
        D[direction][1] * [M[k], M[k]]
    )
    # Last square's bottom-left
    bl_prev = np.min([bl_prev, bl], axis=0)

    # 1/4 circle
    centroid = bl + D_centroid[direction] * [M[k], M[k]]
    low, hight = [M[k], M[k]] * D_X[direction]
    X = np.linspace(low, hight, 100)
    Y = f(X, M[k]) * D_Y[direction]

    # Plot
    color = np.array(create_unique_color_uchar(k)) / 255
    axe.add_patch(patches.Rectangle(bl, width=M[k], height=M[k], fill=False, color=color))
    axe.plot(X + centroid[0], Y + centroid[1], color=color)

    print('{:2d}. {} / {} = {}'.format(k, M[k], M[k-1], M[k] / M[k-1]))
print('Golden ratio: {}'.format((1 + 5**0.5) / 2))

# Show
# lims = np.array([-M[k] - M[k-1], M[k]])
# plt.xlim(lims)
# plt.ylim(-lims[::-1])
axe.set_aspect('equal')
plt.show()
