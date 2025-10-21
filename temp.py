# %%
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize


# def optimise_me(v):


# v = [0.93, 0.1, 2.]
# v = [0.94464603, 0.13938043, 2.75990415]
v = [0.923088, 0.09061906, 2.00789403]


# Example arrays: 560x560, 0.1 cm pitch per pixel
H = W = 560
pixel_pitch_cm = 0.1

# Example transmissions (replace with your real 2D maps aligned at isocentre)
Tx = np.ones((H, W), dtype=np.float32)         # X-jaw (0.01 outside, 1 inside)
Ty = np.ones((H, W), dtype=np.float32)         # Y-jaw (0 outside, 1 inside)
Tmlc = np.ones((H, W), dtype=np.float32)       # MLC (0..1)

# For illustration: add leakage/out-of-field regions
Tx[:80, :] = 0.01; Tx[-80:, :] = 0.0075
Ty[:, :80] = 0.0;  Ty[:, -80:] = 0.0
# Tmlc could be fractional; here leave as ones

T_total = Tx * Ty * Tmlc

# Primary source blur
pri_s = v[0]
pri_x = v[1]
pri_y = v[1]
pri_z = 99.5
sigma_pix_x = pri_x / pixel_pitch_cm
sigma_pix_y = pri_y / pixel_pitch_cm
pri_fluence = gaussian_filter(T_total, sigma=(sigma_pix_x, sigma_pix_y), mode='nearest')

# Secondary source blur
sec_s = 1 - v[0]
sec_x = v[2]
sec_y = v[2]
sec_z = 94.5
sigma_pix_x = sec_x / pixel_pitch_cm
sigma_pix_y = sec_y / pixel_pitch_cm
sec_fluence = gaussian_filter(T_total, sigma=(sigma_pix_x, sigma_pix_y), mode='nearest')

# Beam profile correction filter
oads = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.5, 24.7, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]
factors = [1.000, 1.002, 1.005, 1.005, 1.012, 1.016, 1.013, 1.010, 1.004, 1.000, 1.000, 1.000, 0.995, 0.990, 0.950, 0.800, 0.750, 0.750, 0.250, 0.000, 0.000, 0.000, 0.000, 0.000]
bpc_interp = make_interp_spline(oads, factors, k=1)
x = np.arange(-28, 28, 0.1, dtype=np.float32)
y = np.arange(-28, 28, 0.1, dtype=np.float32)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
bpc = bpc_interp(r)

total_fluence = pri_s * pri_fluence * bpc + sec_s * sec_fluence



rs = np.load("fluence_40x40.npy")
rs[rs < 0] = 0


#     # rs_interp = np.interp(
#     #     np.linspace(-20*np.sqrt(2), 20*np.sqrt(2), 400),
#     #     np.linspace(-28*np.sqrt(2), 28*np.sqrt(2), 560),
#     #     np.diag(rs)
#     # )
#     diff = np.sum(np.abs(total_fluence/np.sum(total_fluence) - rs/np.sum(rs)))
#     print(f"x: {v}, diff: {diff}")
#     return diff * 1e6

# # %%
# x0 = [0.9, 0.101, 2.965]
# bounds = [(0.8, 1.0), (0.01, 1), (1.0, 10.0)]
# result = minimize(optimise_me, x0, bounds=bounds, options={'disp': True})
# print(result)
# print(result.x)
# # # %%

# # %%

# %%
