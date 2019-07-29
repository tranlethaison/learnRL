# Fibonacci spiral with Dynamic Programming style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import create_unique_color_uchar


# Plotting prepare
fig = plt.figure()
axe = fig.add_subplot(111)

# Directions matrix for calculating bottom-left of k-th square
# wrt [[fibo[k-1], fibo[k-1]], [fibo[k], fibo[k]]]
D = np.array(
    [
        [[0, 0], [-1, 0]],  # left
        [[0, 0], [0, -1]],  # down
        [[1, 0], [0, 0]],  # right
        [[0, 1], [0, 0]],  # up
    ]
)

# 1/4 of the circle for k-th square
HalfCircle = lambda x, r: (r ** 2 - x ** 2) ** 0.5
# Directions matrix for calculating centroid
D_centroid = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])  # left  # down  # right  # up
# Directions matrix of X range wrt [r, r]
# ignore point [0, 0] for the sake of simple
D_X = np.array(
    [
        [-1, 0],  # left  => top-left
        [-1, 0],  # down  => bottom-left
        [0, 1],  # right => bottom-right
        [0, 1],  # up    => top-right
    ]
)
# Directions matrix of Y range wrt to Y range
D_Y = np.array(
    [
        1,  # left  => top-left
        -1,  # down  => bottom-left
        -1,  # right => bottom-right
        1,  # up    => top-right
    ]
)

# Init Fibonacci sequence
n = 80
fibo = np.array([None] * (n + 1), np.float128)
fibo[0], fibo[1] = 0, 1

# 1st Fibonacci
k = 1
bl = bl_prev = np.array([0, 0])
color = np.array(create_unique_color_uchar(k)) / 255
axe.add_patch(
    patches.Rectangle(bl, width=fibo[k], height=fibo[k], fill=False, color=color)
)

# k-th Fibonacci
for k in range(2, n + 1):
    fibo[k] = fibo[k - 1] + fibo[k - 2]
    direction = k % 4

    # square's bottom-left
    bl = (
        bl_prev
        + D[direction][0] * [fibo[k - 1], fibo[k - 1]]
        + D[direction][1] * [fibo[k], fibo[k]]
    )

    # 1/4 circle
    centroid = bl + D_centroid[direction] * [fibo[k], fibo[k]]
    low, high = [fibo[k], fibo[k]] * D_X[direction]
    X = np.linspace(low, high, 100)
    Y = HalfCircle(X, fibo[k]) * D_Y[direction]

    # Plot
    color = np.array(create_unique_color_uchar(k)) / 255
    axe.add_patch(
        patches.Rectangle(bl, width=fibo[k], height=fibo[k], fill=False, color=color)
    )
    axe.plot(X + centroid[0], Y + centroid[1], color=color)
    print("{:2d}. {} / {} = {}".format(k, fibo[k], fibo[k - 1], fibo[k] / fibo[k - 1]))

    # Update k-th specific parameters
    bl_prev = np.min([bl_prev, bl], axis=0)

print("Golden ratio: {}".format((1 + 5 ** 0.5) / 2))

# Show
axe.set_aspect("equal")
plt.grid(True)
plt.show()
