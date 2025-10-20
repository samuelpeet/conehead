# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator

# # -----------------------------
# # Geometry setup (mm)
# # -----------------------------
# z_iso = 0.0               # isocentre
# SAD = 1000.0              # Source-Axis Distance
# z_jaws = 500.0            # jaws plane (relative to iso)
# z_primary = 995.0         # primary source plane
# z_secondary = 950.0       # secondary source plane

# # Field size at isocentre (10x10 cm)
# field_half_iso = 50.0

# # # Project field edges from isocentre → jaw plane
# # field_half_jaws = field_half_iso * (z_jaws / (SAD))  # proportional to distance from source

# # Project jaw edges → source planes
# field_half_primary = field_half_iso * (z_primary / z_jaws)
# field_half_secondary = field_half_iso * (z_secondary / z_jaws)

# # -----------------------------
# # Fluence grid setup
# # -----------------------------
# grid_size = 201
# extent = 200.0
# x = np.linspace(-extent, extent, grid_size)
# y = np.linspace(-extent, extent, grid_size)
# xx, yy = np.meshgrid(x, y)

# # -----------------------------
# # Source parameters
# # -----------------------------
# sigma_primary = 1.5    # mm
# sigma_secondary = 25.0  # mm
# strength_primary = 0.9
# strength_secondary = 0.1

# def gaussian_2d(x, y, sigma):
#     return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# def backproject_to_plane(xp, yp, z_plane):
#     """
#     Backprojects a point (xp, yp, z_iso) to an upstream plane at z_plane,
#     assuming linear scaling from iso to the source plane.
#     """
#     scale = (SAD - z_plane) / SAD  # distances measured from source
#     return xp * scale, yp * scale

# # Backproject from iso to each source plane
# x_p, y_p = backproject_to_plane(xx, yy, z_primary)
# x_s, y_s = backproject_to_plane(xx, yy, z_secondary)

# # -----------------------------
# # Apply collimator blocking
# # -----------------------------
# mask_primary = (
#     (np.abs(x_p) <= field_half_primary)
#     & (np.abs(y_p) <= field_half_primary)
# )
# mask_secondary = (
#     (np.abs(x_s) <= field_half_secondary)
#     & (np.abs(y_s) <= field_half_secondary)
# )

# # -----------------------------
# # Compute fluence components
# # -----------------------------
# fluence_primary_src = gaussian_2d(x_p, y_p, sigma_primary)
# fluence_secondary_src = gaussian_2d(x_s, y_s, sigma_secondary)

# # Apply masks
# fluence_primary_src *= mask_primary
# fluence_secondary_src *= mask_secondary

# # # Apply inverse-square law
# # dist_primary = SAD - z_primary
# # dist_secondary = SAD - z_secondary
# # fluence_primary_src /= dist_primary**2
# # fluence_secondary_src /= dist_secondary**2

# # # Combine total fluence
# # fluence_total_src = fluence_primary_src + fluence_secondary_src

# # -----------------------------
# # Forward projection to isocentre
# # -----------------------------
# def forward_project(fluence_src, z_src):
#     """
#     Forward-project fluence from z_src to z_iso=0 plane.
#     Returns interpolated fluence at iso plane grid.
#     """
#     # # Create interpolator
#     # interp = RegularGridInterpolator((y, x), fluence_src, bounds_error=False, fill_value=0.0)

#     # # Scale coordinates from iso plane to source plane
#     # scale = SAD / (SAD - z_src)  # geometric magnification
#     # x_iso = xx * scale
#     # y_iso = yy * scale

#     # # Interpolate
#     # points = np.stack([y_iso.ravel(), x_iso.ravel()], axis=-1)
#     # fluence_at_iso = interp(points).reshape(xx.shape)

#     # Apply inverse-square law
#     dist = SAD / (SAD - z_src)
#     dist = 1.0
#     # fluence_src /= dist**2

#     return fluence_src / dist**2



# # Forward project both sources
# fluence_primary_iso = forward_project(fluence_primary_src, z_primary)
# fluence_secondary_iso = forward_project(fluence_secondary_src, z_secondary)

# # Total fluence at iso
# fluence_total_iso = sigma_primary * fluence_primary_iso + sigma_secondary * fluence_secondary_iso



# # -----------------------------
# # Plot results
# # -----------------------------
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# vmax = fluence_total_iso.max()
# axes[0].imshow(fluence_primary_iso, extent=[-extent, extent, -extent, extent],
#                cmap='inferno')
# axes[0].set_title(f"Primary Source Fluence (z={z_primary} mm)")

# axes[1].imshow(fluence_secondary_iso, extent=[-extent, extent, -extent, extent],
#                cmap='inferno')
# axes[1].set_title(f"Secondary Source Fluence (z={z_secondary} mm)")

# axes[2].imshow(fluence_total_iso, extent=[-extent, extent, -extent, extent],
#                cmap='inferno')
# axes[2].set_title("Total Fluence at Isocentre (z=0)")

# for ax in axes:
#     ax.set_xlabel("x (mm)")
#     ax.set_ylabel("y (mm)")
#     ax.set_aspect('equal')

# plt.tight_layout()
# plt.show()
# # %%


# %%
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

def combine_transmission_maps(Tx_iso, Ty_iso, Tmlc_iso):
    # Elementwise product; all are on the same isocentre grid
    return (Tx_iso.astype(np.float32)
            * Ty_iso.astype(np.float32)
            * Tmlc_iso.astype(np.float32))

def build_coords_for_iso_grid(H, W, pixel_pitch_cm):
    # Center the grid at isocentre (0,0), x right, y up
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    x_coords = (np.arange(W, dtype=np.float64) - cx) * pixel_pitch_cm
    y_coords = (np.arange(H, dtype=np.float64) - cy) * pixel_pitch_cm
    return y_coords, x_coords  # for (row=y, col=x) convention

def preblur_T_iso_per_depth(T_iso, depths_cm, sources, pixel_pitch_cm):
    """
    T_iso: (H,W) combined transmission at isocentre
    depths_cm: sequence of depths d>=0 to evaluate (e.g., unique z-slice depths)
    sources: list of dicts: {'z_s': cm, 'sigma': cm, 'strength': float, 'mu': (cm,cm)}
    Returns dict: blurred[(source_index, depth_index)] -> blurred map (H,W)
    """
    H, W = T_iso.shape
    blurred = {}
    for si, src in enumerate(sources):
        z_s = float(src['z_s'])
        sigma_s = float(src['sigma'])
        for di, d in enumerate(depths_cm):
            s = z_s / (z_s + d)  # projection scale
            sigma_u_cm = sigma_s * s
            sigma_pix = max(sigma_u_cm / pixel_pitch_cm, 1e-6)
            # Separable Gaussian blur; mode='nearest' to avoid wrap artifacts
            B = gaussian_filter(T_iso, sigma=sigma_pix, mode='nearest')
            blurred[(si, di)] = B.astype(np.float32)
    return blurred

def make_interpolator(map2d, y_coords, x_coords):
    # RegularGridInterpolator expects increasing coord arrays; returns bilinear by default
    return RegularGridInterpolator((y_coords, x_coords), map2d, bounds_error=False, fill_value=0.0)

def fluence_two_source_for_slice(
    voxels_xy_cm, d_cm, sources, blurred_maps, y_coords, x_coords, pixel_pitch_cm
):
    """
    voxels_xy_cm: (N,2) array of voxel in-plane coords at this depth slice (cm)
    d_cm: scalar depth for this slice (cm)
    sources: list of dicts: {'z_s','sigma','strength','mu'}
    blurred_maps: dict from preblur_T_iso_per_depth
    Returns: fluence array shape (N,)
    """
    H = len(y_coords); W = len(x_coords)
    flu = np.zeros((voxels_xy_cm.shape[0],), dtype=np.float32)
    for si, src in enumerate(sources):
        z_s = float(src['z_s'])
        sigma_s = float(src['sigma'])
        strength = float(src['strength'])
        mu_s = np.array(src.get('mu', (0.0, 0.0)), dtype=np.float64)

        s = z_s / (z_s + d_cm)  # projection scale
        # Center location on iso plane where to sample blurred map
        # u_mu(v) = s*mu_s + (1 - s)*v
        u_xy = s * mu_s[None, :] + (1.0 - s) * voxels_xy_cm

        # Build/interpolate blurred map for this source/depth
        B = blurred_maps[(si, 0)] if isinstance(blurred_maps, list) else blurred_maps[(si, 0)]
        # If you preblurred per-depth, use the right depth index; here we assume single depth preblur per slice
        # Safer: index by actual depth if you built multiple: blurred_maps[(si, depth_index)]

        interp = make_interpolator(B, y_coords, x_coords)
        vals = interp(u_xy[:, ::-1])  # interpolator takes (y,x)

        # Inverse square; constant for this depth slice (approximation)
        inv_r2 = 1.0 / ((z_s + d_cm) * (z_s + d_cm))
        flu += strength * vals.astype(np.float32) * inv_r2
    return flu


# Example arrays: 500x500, 0.1 cm pitch per pixel
H = W = 500
pixel_pitch_cm = 0.1

# Example transmissions (replace with your real 2D maps aligned at isocentre)
Tx = np.ones((H, W), dtype=np.float32)         # X-jaw (0.01 outside, 1 inside)
Ty = np.ones((H, W), dtype=np.float32)         # Y-jaw (0 outside, 1 inside)
Tmlc = np.ones((H, W), dtype=np.float32)       # MLC (0..1)

# For illustration: add leakage/out-of-field regions
Tx[:, :100] = 0.01; Tx[:, -100:] = 0.01
Ty[:120, :] = 0.0;  Ty[-120:, :] = 0.0
# Tmlc could be fractional; here leave as ones

T_iso = combine_transmission_maps(Tx, Ty, Tmlc)
y_coords, x_coords = build_coords_for_iso_grid(H, W, pixel_pitch_cm)

# Two sources (cm). Match settings.toml [sources_new] if desired.
sources = [
    {"z_s": 99.5,  "sigma": 0.101, "strength": 0.9, "mu": (0.0, 0.0)},  # primary (example)
    {"z_s": 94.5,   "sigma": 2.765, "strength": 0.1, "mu": (0.0, 0.0)},  # secondary (example)
]

# Pick a depth slice and some voxels in that plane
d_cm = 0.0
xs = np.linspace(-20.0, 20.0, 200)
ys = np.zeros_like(xs)
voxels_xy = np.stack([xs, ys], axis=1)

# Preblur once per source/depth (here a single depth; for many depths, loop and cache)
blurred_maps = {}
for si, src in enumerate(sources):
    z_s = src["z_s"]; sigma_s = src["sigma"]
    s = z_s / (z_s + d_cm)
    sigma_u_cm = sigma_s * s
    sigma_pix = max(sigma_u_cm / pixel_pitch_cm, 1e-6)
    B = gaussian_filter(T_iso, sigma=sigma_pix, mode='nearest')
    blurred_maps[(si, 0)] = B.astype(np.float32)

fluence = fluence_two_source_for_slice(voxels_xy, d_cm, sources, blurred_maps,
                                        y_coords, x_coords, pixel_pitch_cm)
print("Fluence along profile:", fluence)
# %%
