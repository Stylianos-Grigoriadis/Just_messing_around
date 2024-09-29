import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



# def lorenz(xyz, *, s=10, r=28, b=2.667):
#     """
#     Parameters
#     ----------
#     xyz : array-like, shape (3,)
#        Point of interest in three-dimensional space.
#     s, r, b : float
#        Parameters defining the Lorenz attractor.
#
#     Returns
#     -------
#     xyz_dot : array, shape (3,)
#        Values of the Lorenz attractor's partial derivatives at *xyz*.
#     """
#     x, y, z = xyz
#     x_dot = s*(y - x)
#     y_dot = r*x - y - x*z
#     z_dot = x*y - b*z
#     return np.array([x_dot, y_dot, z_dot])
#
#
# dt = 0.01
# num_steps = 10000
#
# xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
# xyzs[0] = (0., 1., 1.05)  # Set initial values
# # Step through "time", calculating the partial derivatives at the current point
# # and using them to estimate the next point
# for i in range(num_steps):
#     xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
#
# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot(*xyzs.T, lw=0.5, c='k')
# ax.set_facecolor('white')
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_facecolor('white')  # Axis background color
# fig.patch.set_facecolor('white')  # Figure background color
# ax.set_title("Lorenz Attractor")
#
# ax.grid(False)
#
# # Set the axis background (pane) colors to white
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # RGB + alpha
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#
# # Set the spines (edges) of the panes to be invisible
# ax.xaxis._axinfo['grid'].update(color='white', linestyle='-')
# ax.yaxis._axinfo['grid'].update(color='white', linestyle='-')
# ax.zaxis._axinfo['grid'].update(color='white', linestyle='-')
# plt.show()



# a = 0.2
# b = 0.2
# c = 5.7
# t = 0
# tf = 250
# h = 0.01
#
#
# def derivative(r, t):
#     x = r[0]
#     y = r[1]
#     z = r[2]
#     return np.array([- y - z, x + a * y, b + z * (x - c)])
#
#
# time = np.array([])
# x = np.array([])
# y = np.array([])
# z = np.array([])
# r = np.array([0.1, 0.1, 0.1])
# while (t <= tf):
#     time = np.append(time, t)
#     z = np.append(z, r[2])
#     y = np.append(y, r[1])
#     x = np.append(x, r[0])
#
#     k1 = h * derivative(r, t)
#     k2 = h * derivative(r + k1 / 2, t + h / 2)
#     k3 = h * derivative(r + k2 / 2, t + h / 2)
#     k4 = h * derivative(r + k3, t + h)
#     r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
#
#
#     t = t + h
#
# xyzs = [x,y,z]
#
# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(*xyzs, lw=0.5, c='k')
# ax.set_facecolor('white')
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_facecolor('white')  # Axis background color
# fig.patch.set_facecolor('white')  # Figure background color
# ax.set_title("Rössler attractor")
# ax.grid(False)
# # Set the axis background (pane) colors to white
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # RGB + alpha
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# # Set the spines (edges) of the panes to be invisible
# ax.xaxis._axinfo['grid'].update(color='white', linestyle='-')
# ax.yaxis._axinfo['grid'].update(color='white', linestyle='-')
# ax.zaxis._axinfo['grid'].update(color='white', linestyle='-')
# plt.show()




# Aizawa attractor function
def aizawa(state, a=0.92, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = state
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * x ** 3
    return np.array([dx, dy, dz])

# Set initial conditions and parameters
dt = 0.01
num_steps = 50000
xyzs = np.empty((num_steps + 1, 3))
xyzs[0] = (0.1, 0, 0)  # Initial conditions

# Integrate the Aizawa attractor over time
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + aizawa(xyzs[i]) * dt

# Plotting the Aizawa attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Experimenting with parameters to create "A"-like features
ax.plot(*xyzs.T, lw=0.6, c='k')

# Label axes
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Aizawa Attractor")

# Set background to white
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Remove grid and make panes white
ax.grid(False)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# Show the plot
plt.show()
