#%%
from array import array
import numpy as np

point = np.array([10, 0, 0])

# Gantry and colly zero
source = np.array([0, -100, 0])
plane_x = np.array([1, 0, 0])
plane_y = np.array([0, 1, 0])
plane_z = np.array([0, 0, 1])

# Rotate gantry to 90
theta = 90
phi = (90 - theta) % 360  # IEC 61217
x = 100 * np.cos(phi * np.pi / 180)
y = 100 * -np.sin(phi * np.pi / 180)
z = source[2]
source = np.array([x, y, z])

plane_x = np.array([1, 0, 0])
plane_y = np.array([0, 1, 0])
plane_z = np.array([0, 0, 1])

plane_x_p = np.array([0, 1, 0])
plane_y_p = np.array([-1, 0, 0])
plane_z_p = np.array([0, 0, 1])

R = np.array([plane_x_p, plane_y_p, plane_z_p])
point_p = np.matmul(R, point)


# RR = 2 * np.matmul(plane_y + plane_y_p, (plane_y + plane_y_p).transpose()) / np.matmul((plane_y + plane_y_p).transpose(), plane_y + plane_y_p) - np.eye(3)

v = np.cross(plane_y, plane_y_p)
s = np.linalg.norm(v)
c = np.dot(plane_y, plane_y_p)

v_x = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])

RR = np.eye(3) + v_x + np.dot(v_x, v_x) * 1 / (1 + c)



# %%
