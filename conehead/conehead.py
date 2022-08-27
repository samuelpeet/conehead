import numpy as np
import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "0"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "0";
import numba
from numba import cuda
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import euclidean
import math
from tqdm import tqdm
from .geometry import (
    line_block_plane_collision,
    #line_calc_limit_plane_collision, isocentre_plane_position
)
from .kernel import PolyenergeticKernel
from .dosegrid import DoseGrid
# from .convolve_c import convolve_c # pylint: disable=E0611
from .nist import mu_water



@cuda.jit(device=True)
def cuda_dot(a, b) -> numba.float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def cuda_line_block_plane_collision(pos_plane, ray_start, ray_direction, plane_normal, epsilon):

    plane_point = cuda.local.array(3, numba.float32)
    plane_point[0] = 0
    plane_point[1] = 0
    plane_point[2] = 0

    ndotu = cuda_dot(plane_normal, ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = cuda.local.array(3, numba.float32)
    w[0] = ray_start[0] - plane_point[0]
    w[1] = ray_start[1] - plane_point[1]
    w[2] = ray_start[2] - plane_point[2]

    si = -cuda_dot(plane_normal, w) / ndotu
    pos_plane[0] = w[0] + si * ray_direction[0] + plane_point[0]
    pos_plane[1] = w[1] + si * ray_direction[1] + plane_point[1]
    pos_plane[2] = w[2] + si * ray_direction[2] + plane_point[2]

    return pos_plane


@cuda.jit(device=True)
def cuda_block_transmission(position, block_values) -> numba.float32:

        position[0] = math.floor(position[0] * numba.float32(100))  # Convert tenth of a mm
        position[1] = math.floor(position[1] * numba.float32(100))
        
        position[0] = position[0] + numba.float32(2000)
        position[1] = position[1] + numba.float32(2000)

        # Handle position lying outside the defined blocking area
        for coord in position:
            if coord < 0 or coord > 3999:
                return numba.float32(0)

        transmission = block_values[
            int(position[0])-1,
            int(position[1])-1
        ]
        return transmission


@cuda.jit
def cuda_hit_test(dose_grid_blocked, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_v_y, source_transform, block_values, samples):

    x, y, z = cuda.grid(3)
    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:
        
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        offset = cuda.local.array(3, numba.float32)
        offset[0] = dose_grid_spacing[0] / samples
        offset[1] = dose_grid_spacing[1] / samples
        offset[2] = dose_grid_spacing[2] / samples

        block_factor: numba.float32 = 0
        for ix in range(samples):
            for iy in range(samples):
                for iz in range(samples):
                    
                    # Position of sample
                    pos_sample = cuda.local.array(3, numba.float32)
                    pos_sample[0] = position[0] - dose_grid_spacing[0]/2 + offset[0]/2 + offset[0] * ix
                    pos_sample[1] = position[1] - dose_grid_spacing[1]/2 + offset[1]/2 + offset[1] * iy
                    pos_sample[2] = position[2] - dose_grid_spacing[2]/2 + offset[2]/2 + offset[2] * iz

                    # Determine position on blocking plane in global coords                      
                    ray_direction = cuda.local.array(3, numba.float32)
                    ray_direction[0] = source_position[0] - pos_sample[0]
                    ray_direction[1] = source_position[1] - pos_sample[1]
                    ray_direction[2] = source_position[2] - pos_sample[2]
                    
                    pos_plane = cuda.local.array(3, numba.float32)
                    pos_plane = cuda_line_block_plane_collision(pos_plane, source_position, ray_direction, source_v_y, 1e-6)

                    # Convert to source coords
                    pos_block = cuda.local.array(3, numba.float32)
                    pos_block[0] = cuda_dot(source_transform[0, :], pos_plane)
                    pos_block[1] = cuda_dot(source_transform[1, :], pos_plane)
                    pos_block[2] = cuda_dot(source_transform[2, :], pos_plane)
                   
                    # Reduce to 2D
                    pos_block_2d = cuda.local.array(2, numba.float32)
                    pos_block_2d[0] = pos_block[0]
                    pos_block_2d[1] = pos_block[2]

                    block_factor = block_factor + cuda_block_transmission(pos_block_2d, block_values) / samples**3
        
        dose_grid_blocked[x, y, z] = block_factor


@cuda.jit(device=True)
def cuda_dda_3d(current_voxel, direction, dose_grid_spacing, dose_grid_size, voxels_traversed, intersection_t_values):

    step = cuda.local.array(3, numba.int32)
    step[0] = -1 if direction[0] < 0 else 1
    step[1] = -1 if direction[1] < 0 else 1
    step[2] = -1 if direction[2] < 0 else 1

    if direction[0] < 0:
        direction[0] = -1 * direction[0]
    if direction[1] < 0:
        direction[1] = -1 * direction[1]
    if direction[2] < 0:
        direction[2] = -1 * direction[2]

    t = cuda.local.array(3, numba.float32)
    t[0] = 0.0
    t[1] = 0.0
    t[2] = 0.0
    
    delta_t = cuda.local.array(3, numba.float32)
    delta_t[0] = 0.0
    delta_t[1] = 0.0
    delta_t[2] = 0.0
    big_number = 1000000000.0
    
    if direction[0] == 0.0:
        t[0] = big_number
        delta_t[0] = big_number
    else:
        t[0] = (dose_grid_spacing[0] / 2) / direction[0]
        delta_t[0] = dose_grid_spacing[0] / direction[0]
    if direction[1] == 0.0:
        t[1] = big_number
        delta_t[1] = big_number
    else:
        t[1] = (dose_grid_spacing[1] / 2) / direction[1]
        delta_t[1] = dose_grid_spacing[1] / direction[1]
    if direction[2] == 0.0:
        t[2] = big_number
        delta_t[2] = big_number
    else:
        t[2] = (dose_grid_spacing[2] / 2) / direction[2]
        delta_t[2] = dose_grid_spacing[2] / direction[2]

    xmax = dose_grid_size[0]
    ymax = dose_grid_size[1]
    zmax = dose_grid_size[2]

    intersection_t_values_count = 0
    voxels_traversed_count = 0

    while (current_voxel[0] >= 0 and current_voxel[0] < xmax and
           current_voxel[1] >= 0 and current_voxel[1] < ymax and
           current_voxel[2] >= 0 and current_voxel[2] < zmax):

        voxels_traversed[voxels_traversed_count, 0] = current_voxel[0]
        voxels_traversed[voxels_traversed_count, 1] = current_voxel[1]
        voxels_traversed[voxels_traversed_count, 2] = current_voxel[2]
        voxels_traversed_count += 1
        if t[0] < t[1]:
            if t[0] < t[2]:
                intersection_t_values[intersection_t_values_count] = t[0]
                intersection_t_values_count += 1              
                t[0] += delta_t[0]
                current_voxel[0] += step[0]
            else:
                intersection_t_values[intersection_t_values_count] = t[2]
                intersection_t_values_count += 1
                t[2] += delta_t[2]
                current_voxel[2] += step[2]
        else:
            if t[1] < t[2]:
                intersection_t_values[intersection_t_values_count] = t[1]
                intersection_t_values_count += 1
                t[1] += delta_t[1]
                current_voxel[1] += step[1]
            else:
                intersection_t_values[intersection_t_values_count] = t[2]
                intersection_t_values_count += 1
                t[2] += delta_t[2]
                current_voxel[2] += step[2]

    return (voxels_traversed_count, intersection_t_values_count)


@cuda.jit
def cuda_d_eff(dose_grid_size, dose_grid_origin, dose_grid_spacing, dose_grid_densities, source_position, d_eff):

    x, y, z = cuda.grid(3)
    current_voxel = cuda.local.array(3, numba.int32)
    current_voxel[0] = x
    current_voxel[1] = y
    current_voxel[2] = z

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        max_array_length = 444  # Longest diagonal of 256 x 256 x 256 grid 
        intersection_t_values = cuda.local.array(max_array_length, numba.float32)
        voxels_traversed = cuda.local.array((max_array_length, 3), numba.int32)

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        # Determine direction to source                      
        ray_direction = cuda.local.array(3, numba.float32)
        ray_direction[0] = source_position[0] - position[0]
        ray_direction[1] = source_position[1] - position[1]
        ray_direction[2] = source_position[2] - position[2]
        mag = math.sqrt(ray_direction[0]*ray_direction[0] + ray_direction[1]*ray_direction[1] + ray_direction[2]*ray_direction[2])
        ray_direction[0] /= mag
        ray_direction[1] /= mag
        ray_direction[2] /= mag


        voxels_traversed_count, intersection_t_values_count = cuda_dda_3d(
            current_voxel,
            ray_direction,
            dose_grid_spacing,
            dose_grid_size,
            voxels_traversed,
            intersection_t_values
        )
        d_eff[x, y, z] = 0
        for v in range(voxels_traversed_count):
            if v == 0:
                d_eff[x, y, z] += dose_grid_densities[voxels_traversed[v, 0], voxels_traversed[v, 1], voxels_traversed[v, 2]] * (intersection_t_values[v])    
            else:
                d_eff[x, y, z] += dose_grid_densities[voxels_traversed[v, 0], voxels_traversed[v, 1], voxels_traversed[v, 2]] * (intersection_t_values[v] - intersection_t_values[v - 1])




@cuda.jit(device=True)
def cuda_dist_point_to_line(A, B, C):
    # A is the point, B and C are two points on the line
    # magnitude(cross(A - B, C - B)) / magnitude(C - B). 

    a_b = cuda.local.array(3, numba.float32)
    a_b[0] = A[0] - B[0]    
    a_b[1] = A[1] - B[1]
    a_b[2] = A[2] - B[2]
    a_b_mag = math.sqrt(a_b[0] * a_b[0] + a_b[1] * a_b[1] + a_b[2] * a_b[2])

    c_b = cuda.local.array(3, numba.float32)
    c_b[0] = C[0] - B[0]    
    c_b[1] = C[1] - B[1]
    c_b[2] = C[2] - B[2]
    c_b_mag = math.sqrt(c_b[0] * c_b[0] + c_b[1] * c_b[1] + c_b[2] * c_b[2])

    # Angle between vectors
    theta = math.acos(cuda_dot(a_b, c_b) / (a_b_mag * c_b_mag))
    
    # Cross product
    a_b_cross_c_b = a_b_mag * c_b_mag * math.sin(theta)

    return a_b_cross_c_b / c_b_mag
     


@cuda.jit
def cuda_oad(dose_grid_oad, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_sad, source_transform, source_v_y):

    x, y, z = cuda.grid(3)

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:   

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        # Determine distance/direction to source                      
        distance = cuda.local.array(3, numba.float32)
        distance[0] = source_position[0] - position[0]
        distance[1] = source_position[1] - position[1]
        distance[2] = source_position[2] - position[2]

        # Project position to iso plane 
        pos_plane = cuda.local.array(3, numba.float32)
        pos_plane = cuda_line_block_plane_collision(pos_plane, source_position, distance, source_v_y, 1e-6)
        
        # Convert to source coords
        pos_source = cuda.local.array(3, numba.float32)
        pos_source[0] = cuda_dot(source_transform[0, :], pos_plane)
        pos_source[1] = cuda_dot(source_transform[1, :], pos_plane)
        pos_source[2] = cuda_dot(source_transform[2, :], pos_plane)
        dose_grid_oad[x, y, z] = math.sqrt(pos_source[0] * pos_source[0] + pos_source[2] * pos_source[2])





@cuda.jit
def cuda_fluence(dose_grid_fluence, dose_grid_oad, dose_grid_blocked, dose_grid_size, dose_grid_origin, dose_grid_spacing, source_position, source_sad, sPri, zAnn, sAnn, rInner, rOuter, zExp, sExp, kExp):

    x, y, z = cuda.grid(3)

    if x < dose_grid_size[0] and y < dose_grid_size[1] and z < dose_grid_size[2]:

        # Get voxel position
        position = cuda.local.array(3, numba.float32)
        position[0] = dose_grid_origin[0] + dose_grid_spacing[0] * x
        position[1] = dose_grid_origin[1] + dose_grid_spacing[1] * y
        position[2] = dose_grid_origin[2] + dose_grid_spacing[2] * z

        # Determine distance/direction to source                      
        distance = cuda.local.array(3, numba.float32)
        distance[0] = source_position[0] - position[0]
        distance[1] = source_position[1] - position[1]
        distance[2] = source_position[2] - position[2]
        mag = math.sqrt(
            distance[0] * distance[0] + 
            distance[1] * distance[1] + 
            distance[2] * distance[2]
        )
        oad = dose_grid_oad[x, y, z]

        # Point source
        fluence_point = sPri * math.pow(source_sad / mag, 2)

        # Annular source
        r_ann = oad * zAnn / source_sad
        if r_ann >= rInner and r_ann <= rOuter:
            fluence_ann = sAnn * math.pow(source_sad - zAnn, 2) / math.pow(mag - zAnn, 2)
        else:
            fluence_ann = 0.0
        
        # Exponential source
        oad = 2.0 if oad < 2.0 else oad  # Avoid function blowing up near zero
        r_exp = oad * zExp / source_sad
        fluence_exp = sExp / r_exp * math.exp(-kExp * r_exp) * math.pow(source_sad - zExp, 2) / math.pow(mag - zExp, 2)

        dose_grid_fluence[x, y, z] = (fluence_point + fluence_ann + fluence_exp) * dose_grid_blocked[x, y, z]



class Conehead:

    def calculate(self, source, block, phantom, settings):

        # # Transform phantom to beam coords
        # print("Transforming phantom to beam coords...")
        # transform = Transform(source.position, source.rotation)
        # phantom_beam = np.zeros_like(phantom.positions)
        # _, xlen, ylen, zlen = phantom_beam.shape
        # for x in tqdm(range(xlen)):
        #     for y in range(ylen):
        #         for z in range(zlen):
        #             phantom_beam[:, x, y, z] = transform.global_to_beam(
        #                 phantom.positions[:, x, y, z],
        #             )

        print("Interpolating phantom densities...")
        phantom_densities_interp = RegularGridInterpolator(
            (np.linspace(phantom.origin[0], phantom.origin[0] + phantom.size[0] * phantom.spacing[0], phantom.size[0]),
             np.linspace(phantom.origin[1], phantom.origin[1] + phantom.size[1] * phantom.spacing[1], phantom.size[1]),
             np.linspace(phantom.origin[2], phantom.origin[2] + phantom.size[2] * phantom.spacing[2], phantom.size[2])),
            phantom.densities,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

        # # Create dose grid (just the same size as the phantom for now)
        # self.dose_grid_positions = np.copy(phantom_beam)
        # self.dose_grid_dim = np.array([1, 1, 1], dtype=np.float64)  # cm

        # Create dose grid (just the same size as the phantom for now)
        self.dose_grid = DoseGrid(phantom.size, phantom.origin, phantom.spacing)
        # self.dose_grid_positions = phantom.
        # self.dose_grid_dim = np.array([1, 1, 1], dtype=np.float64) # cm
        # _, xlen, ylen, zlen = self.dose_grid_positions.shape
        # for x in tqdm(range(xlen)):
        #     for y in range(ylen):
        #         for z in range(zlen):
        #             self.dose_grid_positions[:, x, y, z] = transform.global_to_beam(
        #                 self.dose_grid_positions[:, x, y, z],
        #             )




        # Perform hit testing to find which dose grid voxels are in the beam
        print("Performing hit-testing of dose grid voxels...")
        dose_grid_blocked_device = cuda.to_device(np.zeros(self.dose_grid.size))
        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(dose_grid_blocked_device.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(dose_grid_blocked_device.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(dose_grid_blocked_device.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        cuda_hit_test[blockspergrid, threadsperblock](
            dose_grid_blocked_device,
            cuda.to_device(self.dose_grid.size),
            cuda.to_device(self.dose_grid.origin),
            cuda.to_device(self.dose_grid.spacing),
            cuda.to_device(source.position),
            cuda.to_device(source.v_y),
            cuda.to_device(source.transform),
            cuda.to_device(block.block_values),
            settings['fluenceResampling']
        )
        dose_grid_blocked = dose_grid_blocked_device.copy_to_host()

        plt.imshow(dose_grid_blocked[:, 30, :])
        plt.title("Blocked")
        plt.show()





        print("Interpolating densities at points in dose grid...")
        xx, yy, zz = np.meshgrid(
            np.linspace(self.dose_grid.origin[0], self.dose_grid.origin[0] + self.dose_grid.size[0] * self.dose_grid.spacing[0], self.dose_grid.size[0]),
            np.linspace(self.dose_grid.origin[1], self.dose_grid.origin[1] + self.dose_grid.size[1] * self.dose_grid.spacing[1], self.dose_grid.size[1]),
            np.linspace(self.dose_grid.origin[2], self.dose_grid.origin[2] + self.dose_grid.size[2] * self.dose_grid.spacing[2], self.dose_grid.size[2])
        )
        dose_grid_densities = np.float32(phantom_densities_interp((xx, yy, zz)))
        dose_grid_densities_device = cuda.to_device(dose_grid_densities)



        print("Calculating effective depths...")
        dose_grid_densities_device = cuda.to_device(dose_grid_densities)
        dose_grid_d_eff_device = cuda.to_device(np.zeros_like(dose_grid_densities, dtype=np.float32))
        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(dose_grid_densities.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(dose_grid_densities.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(dose_grid_densities.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        cuda_d_eff[blockspergrid, threadsperblock](
            cuda.to_device(self.dose_grid.size),
            cuda.to_device(self.dose_grid.origin),
            cuda.to_device(self.dose_grid.spacing),
            dose_grid_densities_device,
            cuda.to_device(source.position),
            dose_grid_d_eff_device
        )
        dose_grid_d_eff = dose_grid_d_eff_device.copy_to_host()

        plt.imshow(dose_grid_d_eff[30, :, :])
        plt.title("d_eff")
        plt.show()





        print("Calculating off-axis distances")
        dose_grid_oad = np.zeros_like(dose_grid_densities, dtype=np.float32)
        dose_grid_oad_device = cuda.to_device(dose_grid_oad)
        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(dose_grid_oad.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(dose_grid_oad.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(dose_grid_oad.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        cuda_oad[blockspergrid, threadsperblock](
            dose_grid_oad_device,
            cuda.to_device(self.dose_grid.size),
            cuda.to_device(self.dose_grid.origin),
            cuda.to_device(self.dose_grid.spacing),
            cuda.to_device(source.position),
            source.sad,
            cuda.to_device(source.transform),
            cuda.to_device(source.v_y)            
        ) 
        dose_grid_oad = dose_grid_oad_device.copy_to_host()   

        plt.imshow(dose_grid_oad[30, :, :])
        plt.title("OAD")
        plt.show()



        print("Calculating photon fluence...")
        dose_grid_fluence = np.zeros_like(dose_grid_densities, dtype=np.float32)
        dose_grid_fluence_device = cuda.to_device(dose_grid_fluence)
        threadsperblock = (8, 8, 8)
        blockspergrid_x = math.ceil(dose_grid_fluence.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(dose_grid_fluence.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(dose_grid_fluence.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        cuda_fluence[blockspergrid, threadsperblock](
            dose_grid_fluence_device,
            dose_grid_oad_device,
            dose_grid_blocked_device,
            cuda.to_device(self.dose_grid.size),
            cuda.to_device(self.dose_grid.origin),
            cuda.to_device(self.dose_grid.spacing),
            cuda.to_device(source.position),
            source.sad,
            settings['sPri'],
            settings['zAnn'],
            settings['sAnn'],
            settings['rInner'],
            settings['rOuter'],
            settings['zExp'],
            settings['sExp'],
            settings['kExp']
        )
        dose_grid_fluence = dose_grid_fluence_device.copy_to_host()

        plt.imshow(dose_grid_fluence[30, :, :])
        plt.title("Fluence")
        plt.show()





        # Calculate beam softening factor for dose grid voxels
        print("Calculating beam softening factor...")
        f_soften = np.ones_like(dose_grid_oad)
        f_soften[dose_grid_oad < settings['softLimit']] = 1 / (
            1 - settings['softRatio'] *
            dose_grid_oad[dose_grid_oad < settings['softLimit']]
        )

        plt.imshow(f_soften[30, :, :])
        plt.title("Beam softening")
        plt.show()



        # Calculate beam softening factor for dose grid voxels
        print("Calculating horn factor...")
        f_horn = np.ones_like(dose_grid_oad)
        f_horn += dose_grid_oad * settings['hornRatio']

        plt.imshow(f_horn[30, :, :])
        plt.title("Horn factor")
        plt.show()




        # # Calculate TERMA of dose grid voxels
        # print("Calculating TERMA...")
        # E = np.linspace(settings['eLow'], settings['eHigh'], settings['eNum'])
        # spectrum_weights = source.weights(E)
        # mu_w = mu_water(E)
        # self.dose_grid_terma = np.zeros_like(dose_grid_blocked)
        # xlen, ylen, zlen = self.dose_grid_terma.shape
        # for x in tqdm(range(xlen)):
        #     for y in range(ylen):
        #         for z in range(zlen):
        #             if not dose_grid_blocked[x, y, z]:
        #                 continue
        #             self.dose_grid_terma[x, y, z] = (
        #                 np.sum(
        #                     spectrum_weights *
        #                     self.dose_grid_fluence[x, y, z] *
        #                     np.exp(
        #                         -mu_w * f_soften[x, y, z] *
        #                         dose_grid_d_eff[x, y, z]
        #                     ) * E * mu_w
        #                 ) * f_horn[x, y, z]
        #             )

        # # Calculate dose of dose grid voxels
        # print("Convolving kernel...")
        # kernel = PolyenergeticKernel()
        # dose_grid_dose = np.zeros_like(self.dose_grid_terma, dtype=np.float64)
        # phis = np.array(
        #     sorted([p for p in kernel.cumulative.keys() if p != "radii"]),
        #     dtype=np.float64
        # )
        # thetas = np.linspace(0, 360, 12, endpoint=False, dtype=np.float64)
        # convolve_c(
        #     self.dose_grid_terma,
        #     dose_grid_dose,
        #     self.dose_grid_dim,
        #     thetas,
        #     phis,
        #     kernel
        # )
        # self.dose_grid_dose = dose_grid_dose

    def _in_annulus(self, r, R_inner, R_outer):
        """Check if point with distance r from origin lies inside annular
        boundaries.

        Parameters
        ----------
        r : float
            Distance of point from origin in annular plane
        R_inner : float
            Inner radius of annulus
        R_outer : float
            Outer radius of annulus

        Returns
        -------
        float
            1.0 if inside annulus, 1.0 if outside annulus
        """
        inside = np.zeros_like(r)
        inside[(r > R_inner) & (r < R_outer)] = 1.0
        return inside

    def plot(self):
        # Plotting for debug purposes
        print("Calculation complete. Now plotting...")

        def plot_fluence():  # pylint: disable=W0612
            f1 = plt.figure()  # pylint: disable=W0612
            ax1 = plt.gca()
            im = ax1.imshow(  # pylint: disable=W0612
                np.rot90(self.dose_grid_fluence[:, 20, :]),
                extent=[-20.5, 20.5, -40.5, .5],
                aspect='equal',
                cmap='viridis'
            )
            # Minor ticks
            ax1.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
            ax1.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
            # ax1.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
            ax1.set_title('Fluence')
            # plt.colorbar(im)

        def plot_terma():  # pylint: disable=W0612
            f2 = plt.figure()  # pylint: disable=W0612
            ax2 = plt.gca()
            im = ax2.imshow(  # pylint: disable=W0612
                np.rot90(self.dose_grid_terma[:, 20, :]),
                extent=[-20.5, 20.5, -40.5, .5],
                aspect='equal',
                cmap='viridis'
            )
            # Minor ticks
            ax2.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
            ax2.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
            # ax2.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
            ax2.set_title('TERMA')
            # plt.colorbar(im)

        def plot_dose():
            f3 = plt.figure()  # pylint: disable=W0612
            ax3 = plt.gca()
            im = ax3.imshow(  # pylint: disable=W0612
                np.rot90(self.dose_grid_dose[:, 20, :]),
                extent=[-20.5, 20.5, -40.5, .5],
                aspect='equal',
                cmap='viridis'
            )
            ax3.set_xlim([-20.5, 20.5])
            ax3.set_ylim([-40.5, 0.5])
            # Minor ticks
            ax3.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
            ax3.set_yticks(np.arange(-39.5, 0, 1.0), minor=True)
            # ax3.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
            ax3.set_title('Dose')
            # plt.colorbar(im)

            f6 = plt.figure()  # pylint: disable=W0612
            ax6 = plt.gca()
            im = ax6.imshow(
                self.dose_grid_dose[:, :, 30],
                extent=[-20.5, 20.5, -20.5, 20.5],
                aspect='equal',
                cmap='viridis'
            )
            ax6.set_xlim([-20.5, 20.5])
            ax6.set_ylim([-20.5, 20.5])
            # Minor ticks
            ax6.set_xticks(np.arange(-19.5, 20.0, 1.0), minor=True)
            ax6.set_yticks(np.arange(-19.5, 20.0, 1.0), minor=True)
            # ax6.grid(which="minor", color="#666666", linestyle='-', linewidth=1)
            ax6.set_title('Dose')
            # plt.colorbar(im)

        def plot_30x30_profiles():  # pylint: disable=W0612
            f4 = plt.figure()  # pylint: disable=W0612
            ax4 = plt.gca()
            ax4.plot(
                self.dose_grid_positions[0, :, 20, 35],
                self.dose_grid_dose[:, 20, 35] / self.dose_grid_dose[20, 20, 35] * 100,
                'or',
                # (self.dose_grid_dose[:, 20, 38] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='5 cm calc'
            )
            x5 = 0.1 * np.array([-189, -186.9, -184.8, -182.7, -180.6, -178.5, -176.4, -174.3, -172.2, -170.1, -168, -166.95, -165.9, -164.85, -163.8, -162.75, -161.7, -160.65, -159.6, -158.55, -157.5, -156.45, -155.4, -154.35, -153.3, -152.25, -151.2, -150.15, -149.1, -148.05, -147, -144.9, -142.8, -140.7, -138.6, -136.5, -134.4, -132.3, -130.2, -128.1, -126, -123.9, -121.8, -119.7, -117.6, -115.5, -113.4, -111.3, -109.2, -107.1, -105, -102.9, -100.8, -98.7, -96.6, -94.5, -92.4, -90.3, -88.2, -86.1, -84, -81.9, -79.8, -77.7, -75.6, -73.5, -71.4, -69.3, -67.2, -65.1, -63, -60.9, -58.8, -56.7, -54.6, -52.5, -50.4, -48.3, -46.2, -44.1, -42, -39.9, -37.8, -35.7, -33.6, -31.5, -29.4, -27.3, -25.2, -23.1, -21, -18.9, -16.8, -14.7, -12.6, -10.5, -8.4, -6.3, -4.2, -2.1, 0, 2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21, 23.1, 25.2, 27.3, 29.4, 31.5, 33.6, 35.7, 37.8, 39.9, 42, 44.1, 46.2, 48.3, 50.4, 52.5, 54.6, 56.7, 58.8, 60.9, 63, 65.1, 67.2, 69.3, 71.4, 73.5, 75.6, 77.7, 79.8, 81.9, 84, 86.1, 88.2, 90.3, 92.4, 94.5, 96.6, 98.7, 100.8, 102.9, 105, 107.1, 109.2, 111.3, 113.4, 115.5, 117.6, 119.7, 121.8, 123.9, 126, 128.1, 130.2, 132.3, 134.4, 136.5, 138.6, 140.7, 142.8, 144.9, 147, 148.05, 149.1, 150.15, 151.2, 152.25, 153.3, 154.35, 155.4, 156.45, 157.5, 158.55, 159.6, 160.65, 161.7, 162.75, 163.8, 164.85, 165.9, 166.95, 168, 170.1, 172.2, 174.3, 176.4, 178.5, 180.6, 182.7, 184.8, 186.9, 189])
            d5 = np.array([2.32E-03, 2.36E-03, 2.55E-03, 2.62E-03, 2.83E-03, 2.93E-03, 3.11E-03, 3.19E-03, 3.47E-03, 3.62E-03, 3.89E-03, 4.04E-03, 4.15E-03, 4.35E-03, 4.52E-03, 4.73E-03, 5.07E-03, 5.50E-03, 6.33E-03, 8.43E-03, 1.34E-02, 2.17E-02, 2.81E-02, 3.10E-02, 3.18E-02, 3.24E-02, 3.30E-02, 3.30E-02, 3.33E-02, 3.34E-02, 3.38E-02, 3.39E-02, 3.40E-02, 3.42E-02, 3.43E-02, 3.44E-02, 3.45E-02, 3.45E-02, 3.45E-02, 3.49E-02, 3.48E-02, 3.50E-02, 3.48E-02, 3.50E-02, 3.52E-02, 3.50E-02, 3.51E-02, 3.51E-02, 3.51E-02, 3.50E-02, 3.54E-02, 3.52E-02, 3.53E-02, 3.52E-02, 3.52E-02, 3.50E-02, 3.52E-02, 3.53E-02, 3.53E-02, 3.52E-02, 3.52E-02, 3.54E-02, 3.52E-02, 3.51E-02, 3.53E-02, 3.52E-02, 3.54E-02, 3.51E-02, 3.52E-02, 3.53E-02, 3.52E-02, 3.53E-02, 3.52E-02, 3.50E-02, 3.52E-02, 3.51E-02, 3.51E-02, 3.51E-02, 3.50E-02, 3.50E-02, 3.50E-02, 3.49E-02, 3.49E-02, 3.49E-02, 3.47E-02, 3.48E-02, 3.47E-02, 3.46E-02, 3.46E-02, 3.46E-02, 3.46E-02, 3.45E-02, 3.45E-02, 3.44E-02, 3.46E-02, 3.42E-02, 3.44E-02, 3.42E-02, 3.44E-02, 3.43E-02, 3.43E-02, 3.45E-02, 3.42E-02, 3.42E-02, 3.46E-02, 3.43E-02, 3.46E-02, 3.45E-02, 3.46E-02, 3.45E-02, 3.45E-02, 3.46E-02, 3.46E-02, 3.48E-02, 3.45E-02, 3.46E-02, 3.47E-02, 3.46E-02, 3.47E-02, 3.49E-02, 3.50E-02, 3.49E-02, 3.50E-02, 3.50E-02, 3.49E-02, 3.49E-02, 3.50E-02, 3.52E-02, 3.51E-02, 3.53E-02, 3.51E-02, 3.51E-02, 3.54E-02, 3.51E-02, 3.51E-02, 3.52E-02, 3.55E-02, 3.54E-02, 3.53E-02, 3.51E-02, 3.52E-02, 3.52E-02, 3.54E-02, 3.53E-02, 3.53E-02, 3.52E-02, 3.54E-02, 3.53E-02, 3.51E-02, 3.52E-02, 3.51E-02, 3.53E-02, 3.51E-02, 3.52E-02, 3.52E-02, 3.50E-02, 3.50E-02, 3.53E-02, 3.49E-02, 3.51E-02, 3.49E-02, 3.49E-02, 3.48E-02, 3.48E-02, 3.46E-02, 3.46E-02, 3.47E-02, 3.44E-02, 3.41E-02, 3.40E-02, 3.38E-02, 3.37E-02, 3.36E-02, 3.36E-02, 3.32E-02, 3.28E-02, 3.24E-02, 3.15E-02, 2.97E-02, 2.48E-02, 1.67E-02, 1.00E-02, 7.03E-03, 5.86E-03, 5.27E-03, 4.95E-03, 4.69E-03, 4.50E-03, 4.33E-03, 4.19E-03, 4.08E-03, 3.79E-03, 3.63E-03, 3.43E-03, 3.25E-03, 3.08E-03, 2.99E-03, 2.80E-03, 2.70E-03, 2.50E-03, 2.56E-03])
            d5 = d5 / d5[100] * 100
            ax4.plot(x5, d5, 'r', markersize=5, label='5 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 30],
                self.dose_grid_dose[:, 20, 30] / self.dose_grid_dose[20, 20, 30] * 100,
                'ob',
                # (self.dose_grid_dose[:, 20, 30] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='10 cm calc'
            )
            x10 =	0.1 * np.array([-198.00, -195.80, -193.60, -191.40, -189.20, -187.00, -184.80, -182.60, -180.40, -178.20, -176.00, -174.90, -173.80, -172.70, -171.60, -170.50, -169.40, -168.30, -167.20, -166.10, -165.00, -163.90, -162.80, -161.70, -160.60, -159.50, -158.40, -157.30, -156.20, -155.10, -154.00, -151.80, -149.60, -147.40, -145.20, -143.00, -140.80, -138.60, -136.40, -134.20, -132.00, -129.80, -127.60, -125.40, -123.20, -121.00, -118.80, -116.60, -114.40, -112.20, -110.00, -107.80, -105.60, -103.40, -101.20, -99.00, -96.80, -94.60, -92.40, -90.20, -88.00, -85.80, -83.60, -81.40, -79.20, -77.00, -74.80, -72.60, -70.40, -68.20, -66.00, -63.80, -61.60, -59.40, -57.20, -55.00, -52.80, -50.60, -48.40, -46.20, -44.00, -41.80, -39.60, -37.40, -35.20, -33.00, -30.80, -28.60, -26.40, -24.20, -22.00, -19.80, -17.60, -15.40, -13.20, -11.00, -8.80, -6.60, -4.40, -2.20, 0.00, 2.20, 4.40, 6.60, 8.80, 11.00, 13.20, 15.40, 17.60, 19.80, 22.00, 24.20, 26.40, 28.60, 30.80, 33.00, 35.20, 37.40, 39.60, 41.80, 44.00, 46.20, 48.40, 50.60, 52.80, 55.00, 57.20, 59.40, 61.60, 63.80, 66.00, 68.20, 70.40, 72.60, 74.80, 77.00, 79.20, 81.40, 83.60, 85.80, 88.00, 90.20, 92.40, 94.60, 96.80, 99.00, 101.20, 103.40, 105.60, 107.80, 110.00, 112.20, 114.40, 116.60, 118.80, 121.00, 123.20, 125.40, 127.60, 129.80, 132.00, 134.20, 136.40, 138.60, 140.80, 143.00, 145.20, 147.40, 149.60, 151.80, 154.00, 155.10, 156.20, 157.30, 158.40, 159.50, 160.60, 161.70, 162.80, 163.90, 165.00, 166.10, 167.20, 168.30, 169.40, 170.50, 171.60, 172.70, 173.80, 174.90, 176.00, 178.20, 180.40, 182.60, 184.80, 187.00, 189.20, 191.40, 193.60, 195.80, 198.00])
            d10 = np.array([2.64E-03, 2.71E-03, 2.85E-03, 2.94E-03, 3.11E-03, 3.28E-03, 3.43E-03, 3.58E-03, 3.80E-03, 3.99E-03, 4.15E-03, 4.38E-03, 4.50E-03, 4.55E-03, 4.72E-03, 4.91E-03, 5.16E-03, 5.50E-03, 6.00E-03, 7.09E-03, 9.85E-03, 1.50E-02, 2.03E-02, 2.30E-02, 2.42E-02, 2.47E-02, 2.50E-02, 2.53E-02, 2.53E-02, 2.55E-02, 2.58E-02, 2.60E-02, 2.61E-02, 2.64E-02, 2.64E-02, 2.68E-02, 2.70E-02, 2.69E-02, 2.70E-02, 2.72E-02, 2.73E-02, 2.73E-02, 2.75E-02, 2.74E-02, 2.78E-02, 2.76E-02, 2.76E-02, 2.77E-02, 2.77E-02, 2.79E-02, 2.78E-02, 2.81E-02, 2.80E-02, 2.81E-02, 2.82E-02, 2.82E-02, 2.81E-02, 2.82E-02, 2.80E-02, 2.83E-02, 2.82E-02, 2.84E-02, 2.83E-02, 2.82E-02, 2.82E-02, 2.84E-02, 2.84E-02, 2.84E-02, 2.83E-02, 2.84E-02, 2.83E-02, 2.83E-02, 2.85E-02, 2.83E-02, 2.85E-02, 2.84E-02, 2.84E-02, 2.86E-02, 2.83E-02, 2.82E-02, 2.85E-02, 2.82E-02, 2.82E-02, 2.83E-02, 2.82E-02, 2.81E-02, 2.83E-02, 2.81E-02, 2.83E-02, 2.80E-02, 2.81E-02, 2.83E-02, 2.82E-02, 2.80E-02, 2.80E-02, 2.81E-02, 2.80E-02, 2.81E-02, 2.80E-02, 2.78E-02, 2.79E-02, 2.79E-02, 2.79E-02, 2.81E-02, 2.79E-02, 2.81E-02, 2.82E-02, 2.81E-02, 2.79E-02, 2.82E-02, 2.83E-02, 2.81E-02, 2.82E-02, 2.81E-02, 2.81E-02, 2.79E-02, 2.82E-02, 2.80E-02, 2.81E-02, 2.81E-02, 2.79E-02, 2.83E-02, 2.83E-02, 2.83E-02, 2.83E-02, 2.85E-02, 2.83E-02, 2.83E-02, 2.85E-02, 2.85E-02, 2.84E-02, 2.83E-02, 2.84E-02, 2.83E-02, 2.84E-02, 2.84E-02, 2.83E-02, 2.83E-02, 2.83E-02, 2.82E-02, 2.82E-02, 2.83E-02, 2.82E-02, 2.83E-02, 2.82E-02, 2.81E-02, 2.81E-02, 2.80E-02, 2.79E-02, 2.80E-02, 2.80E-02, 2.79E-02, 2.78E-02, 2.77E-02, 2.78E-02, 2.78E-02, 2.77E-02, 2.76E-02, 2.75E-02, 2.74E-02, 2.74E-02, 2.72E-02, 2.71E-02, 2.70E-02, 2.70E-02, 2.68E-02, 2.67E-02, 2.65E-02, 2.63E-02, 2.61E-02, 2.59E-02, 2.58E-02, 2.57E-02, 2.56E-02, 2.53E-02, 2.49E-02, 2.48E-02, 2.41E-02, 2.31E-02, 2.03E-02, 1.51E-02, 9.85E-03, 7.17E-03, 6.01E-03, 5.55E-03, 5.24E-03, 5.09E-03, 4.84E-03, 4.73E-03, 4.58E-03, 4.48E-03, 4.24E-03, 4.02E-03, 3.81E-03, 3.65E-03, 3.50E-03, 3.31E-03, 3.19E-03, 3.01E-03, 2.88E-03, 2.77E-03])
            d10 = d10 / d10[100] * 100
            ax4.plot(x10, d10, 'b', markersize=5, label='10 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 20],
                self.dose_grid_dose[:, 20, 20] / self.dose_grid_dose[20, 20, 20] * 100,
                'og',
                # (self.dose_grid_dose[:, 20, 10] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='20 cm calc'
            )
            x20 = 0.1 * np.array([-216, -213.6, -211.2, -208.8, -206.4, -204, -201.6, -199.2, -196.8, -194.4, -192, -190.8, -189.6, -188.4, -187.2, -186, -184.8, -183.6, -182.4, -181.2, -180, -178.8, -177.6, -176.4, -175.2, -174, -172.8, -171.6, -170.4, -169.2, -168, -165.6, -163.2, -160.8, -158.4, -156, -153.6, -151.2, -148.8, -146.4, -144, -141.6, -139.2, -136.8, -134.4, -132, -129.6, -127.2, -124.8, -122.4, -120, -117.6, -115.2, -112.8, -110.4, -108, -105.6, -103.2, -100.8, -98.4, -96, -93.6, -91.2, -88.8, -86.4, -84, -81.6, -79.2, -76.8, -74.4, -72, -69.6, -67.2, -64.8, -62.4, -60, -57.6, -55.2, -52.8, -50.4, -48, -45.6, -43.2, -40.8, -38.4, -36, -33.6, -31.2, -28.8, -26.4, -24, -21.6, -19.2, -16.8, -14.4, -12, -9.6, -7.2, -4.8, -2.4, 0, 2.4, 4.8, 7.2, 9.6, 12, 14.4, 16.8, 19.2, 21.6, 24, 26.4, 28.8, 31.2, 33.6, 36, 38.4, 40.8, 43.2, 45.6, 48, 50.4, 52.8, 55.2, 57.6, 60, 62.4, 64.8, 67.2, 69.6, 72, 74.4, 76.8, 79.2, 81.6, 84, 86.4, 88.8, 91.2, 93.6, 96, 98.4, 100.8, 103.2, 105.6, 108, 110.4, 112.8, 115.2, 117.6, 120, 122.4, 124.8, 127.2, 129.6, 132, 134.4, 136.8, 139.2, 141.6, 144, 146.4, 148.8, 151.2, 153.6, 156, 158.4, 160.8, 163.2, 165.6, 168, 169.2, 170.4, 171.6, 172.8, 174, 175.2, 176.4, 177.6, 178.8, 180, 181.2, 182.4, 183.6, 184.8, 186, 187.2, 188.4, 189.6, 190.8, 192, 194.4, 196.8, 199.2, 201.6, 204, 206.4, 208.8, 211.2, 213.6, 216])
            d20 = np.array([2.31E-03, 2.40E-03, 2.48E-03, 2.62E-03, 2.73E-03, 2.84E-03, 2.92E-03, 3.10E-03, 3.20E-03, 3.35E-03, 3.50E-03, 3.57E-03, 3.68E-03, 3.73E-03, 3.84E-03, 3.93E-03, 4.06E-03, 4.26E-03, 4.55E-03, 5.14E-03, 6.56E-03, 9.18E-03, 1.18E-02, 1.33E-02, 1.38E-02, 1.41E-02, 1.43E-02, 1.46E-02, 1.47E-02, 1.47E-02, 1.48E-02, 1.51E-02, 1.51E-02, 1.53E-02, 1.54E-02, 1.56E-02, 1.57E-02, 1.59E-02, 1.59E-02, 1.60E-02, 1.61E-02, 1.62E-02, 1.64E-02, 1.65E-02, 1.65E-02, 1.65E-02, 1.65E-02, 1.66E-02, 1.68E-02, 1.68E-02, 1.68E-02, 1.69E-02, 1.70E-02, 1.71E-02, 1.70E-02, 1.71E-02, 1.72E-02, 1.73E-02, 1.72E-02, 1.73E-02, 1.73E-02, 1.74E-02, 1.74E-02, 1.74E-02, 1.75E-02, 1.75E-02, 1.75E-02, 1.75E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.77E-02, 1.76E-02, 1.78E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.76E-02, 1.78E-02, 1.77E-02, 1.76E-02, 1.77E-02, 1.78E-02, 1.78E-02, 1.78E-02, 1.78E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.77E-02, 1.79E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.76E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.76E-02, 1.75E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.77E-02, 1.76E-02, 1.78E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.76E-02, 1.78E-02, 1.77E-02, 1.77E-02, 1.77E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.76E-02, 1.77E-02, 1.75E-02, 1.75E-02, 1.75E-02, 1.75E-02, 1.73E-02, 1.75E-02, 1.74E-02, 1.73E-02, 1.73E-02, 1.73E-02, 1.71E-02, 1.70E-02, 1.70E-02, 1.70E-02, 1.70E-02, 1.70E-02, 1.68E-02, 1.69E-02, 1.67E-02, 1.67E-02, 1.66E-02, 1.66E-02, 1.65E-02, 1.63E-02, 1.63E-02, 1.61E-02, 1.61E-02, 1.60E-02, 1.60E-02, 1.59E-02, 1.56E-02, 1.57E-02, 1.55E-02, 1.54E-02, 1.52E-02, 1.51E-02, 1.48E-02, 1.47E-02, 1.46E-02, 1.46E-02, 1.44E-02, 1.42E-02, 1.40E-02, 1.33E-02, 1.19E-02, 9.18E-03, 6.63E-03, 5.18E-03, 4.63E-03, 4.30E-03, 4.18E-03, 4.01E-03, 3.94E-03, 3.89E-03, 3.77E-03, 3.67E-03, 3.49E-03, 3.37E-03, 3.19E-03, 3.11E-03, 2.95E-03, 2.86E-03, 2.72E-03, 2.63E-03, 2.57E-03, 2.51E-03])
            d20 = d20 / d20[101] * 100
            ax4.plot(x20, d20, 'g', markersize=5, label='20 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 10],
                self.dose_grid_dose[:, 20, 10] / self.dose_grid_dose[20, 20, 10] * 100,
                'om',
                # (self.dose_grid_dose[:, 20, 10] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='30 cm calc'
            )
            x30 = 0.1 * np.array([-234, -231.4, -228.8, -226.2, -223.6, -221, -218.4, -215.8, -213.2, -210.6, -208, -206.7, -205.4, -204.1, -202.8, -201.5, -200.2, -198.9, -197.6, -196.3, -195, -193.7, -192.4, -191.1, -189.8, -188.5, -187.2, -185.9, -184.6, -183.3, -182, -179.4, -176.8, -174.2, -171.6, -169, -166.4, -163.8, -161.2, -158.6, -156, -153.4, -150.8, -148.2, -145.6, -143, -140.4, -137.8, -135.2, -132.6, -130, -127.4, -124.8, -122.2, -119.6, -117, -114.4, -111.8, -109.2, -106.6, -104, -101.4, -98.8, -96.2, -93.6, -91, -88.4, -85.8, -83.2, -80.6, -78, -75.4, -72.8, -70.2, -67.6, -65, -62.4, -59.8, -57.2, -54.6, -52, -49.4, -46.8, -44.2, -41.6, -39, -36.4, -33.8, -31.2, -28.6, -26, -23.4, -20.8, -18.2, -15.6, -13, -10.4, -7.8, -5.2, -2.6, 0, 2.6, 5.2, 7.8, 10.4, 13, 15.6, 18.2, 20.8, 23.4, 26, 28.6, 31.2, 33.8, 36.4, 39, 41.6, 44.2, 46.8, 49.4, 52, 54.6, 57.2, 59.8, 62.4, 65, 67.6, 70.2, 72.8, 75.4, 78, 80.6, 83.2, 85.8, 88.4, 91, 93.6, 96.2, 98.8, 101.4, 104, 106.6, 109.2, 111.8, 114.4, 117, 119.6, 122.2, 124.8, 127.4, 130, 132.6, 135.2, 137.8, 140.4, 143, 145.6, 148.2, 150.8, 153.4, 156, 158.6, 161.2, 163.8, 166.4, 169, 171.6, 174.2, 176.8, 179.4, 182, 183.3, 184.6, 185.9, 187.2, 188.5, 189.8, 191.1, 192.4, 193.7, 195, 196.3, 197.6, 198.9, 200.2, 201.5, 202.8, 204.1, 205.4, 206.7, 208, 210.6, 213.2, 215.8])
            d30 = np.array([1.64E-03, 1.70E-03, 1.77E-03, 1.86E-03, 1.91E-03, 1.97E-03, 2.06E-03, 2.15E-03, 2.21E-03, 2.32E-03, 2.42E-03, 2.45E-03, 2.48E-03, 2.53E-03, 2.60E-03, 2.64E-03, 2.76E-03, 2.79E-03, 2.93E-03, 3.23E-03, 3.80E-03, 4.90E-03, 6.50E-03, 7.47E-03, 7.95E-03, 8.19E-03, 8.31E-03, 8.35E-03, 8.43E-03, 8.43E-03, 8.62E-03, 8.66E-03, 8.78E-03, 8.89E-03, 9.00E-03, 9.04E-03, 9.12E-03, 9.14E-03, 9.26E-03, 9.45E-03, 9.44E-03, 9.53E-03, 9.64E-03, 9.61E-03, 9.72E-03, 9.72E-03, 9.81E-03, 9.85E-03, 9.88E-03, 9.95E-03, 1.01E-02, 1.00E-02, 1.02E-02, 1.01E-02, 1.02E-02, 1.02E-02, 1.02E-02, 1.03E-02, 1.03E-02, 1.04E-02, 1.04E-02, 1.04E-02, 1.05E-02, 1.04E-02, 1.06E-02, 1.06E-02, 1.06E-02, 1.06E-02, 1.07E-02, 1.07E-02, 1.06E-02, 1.07E-02, 1.08E-02, 1.08E-02, 1.08E-02, 1.08E-02, 1.09E-02, 1.08E-02, 1.08E-02, 1.09E-02, 1.08E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.08E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.10E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.10E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.08E-02, 1.09E-02, 1.09E-02, 1.10E-02, 1.09E-02, 1.09E-02, 1.09E-02, 1.08E-02, 1.09E-02, 1.09E-02, 1.08E-02, 1.09E-02, 1.08E-02, 1.08E-02, 1.08E-02, 1.09E-02, 1.08E-02, 1.08E-02, 1.09E-02, 1.08E-02, 1.08E-02, 1.07E-02, 1.07E-02, 1.07E-02, 1.07E-02, 1.07E-02, 1.06E-02, 1.06E-02, 1.05E-02, 1.06E-02, 1.06E-02, 1.05E-02, 1.05E-02, 1.04E-02, 1.04E-02, 1.04E-02, 1.04E-02, 1.02E-02, 1.03E-02, 1.02E-02, 1.01E-02, 1.01E-02, 1.01E-02, 1.00E-02, 9.96E-03, 9.96E-03, 9.89E-03, 9.80E-03, 9.80E-03, 9.75E-03, 9.61E-03, 9.57E-03, 9.62E-03, 9.52E-03, 9.37E-03, 9.36E-03, 9.20E-03, 9.17E-03, 9.10E-03, 8.96E-03, 8.94E-03, 8.81E-03, 8.73E-03, 8.65E-03, 8.58E-03, 8.56E-03, 8.44E-03, 8.41E-03, 8.25E-03, 8.14E-03, 7.85E-03, 7.28E-03, 6.11E-03, 4.64E-03, 3.63E-03, 3.18E-03, 2.97E-03, 2.83E-03, 2.74E-03, 2.66E-03, 2.63E-03, 2.55E-03, 2.56E-03, 2.44E-03, 2.32E-03, 2.24E-03])
            d30 = d30 / d30[101] * 100
            ax4.plot(x30, d30, 'm', markersize=5, label='30 cm tank')

            ax4.set_xlim([-25, 25])
            ax4.set_ylim([0, 110])
            ax4.set_title("Profiles")
            ax4.set_xlabel("Position [cm]")
            ax4.set_ylabel("Relative Value [%]")
            # x0, x1 = ax4.get_xlim()
            # y0, y1 = ax4.get_ylim()
            # ax4.set_aspect(abs(x1-x0)/abs(y1-y0))
            ax4.legend()

        def plot_10x10_profiles():  # pylint: disable=W0612
            f4 = plt.figure()  # pylint: disable=W0612
            ax4 = plt.gca()
            ax4.plot(
                self.dose_grid_positions[0, :, 20, 35],
                self.dose_grid_dose[:, 20, 35] / self.dose_grid_dose[20, 20, 35] * 100,
                # (self.dose_grid_dose[:, 20, 38] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='5 cm calc'
            )
            x5 = 0.1 * np.array([-84, -81.9, -79.8, -77.7, -75.6, -73.5, -71.4, -69.3, -67.2, -65.1, -63, -61.95, -60.9, -59.85, -58.8, -57.75, -56.7, -55.65, -54.6, -53.55, -52.5, -51.45, -50.4, -49.35, -48.3, -47.25, -46.2, -45.15, -44.1, -43.05, -42, -39.9, -37.8, -35.7, -33.6, -31.5, -29.4, -27.3, -25.2, -23.1, -21, -18.9, -16.8, -14.7, -12.6, -10.5, -8.4, -6.3, -4.2, -2.1, 0, 2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21, 23.1, 25.2, 27.3, 29.4, 31.5, 33.6, 35.7, 37.8, 39.9, 42, 43.05, 44.1, 45.15, 46.2, 47.25, 48.3, 49.35, 50.4, 51.45, 52.5, 53.55, 54.6, 55.65, 56.7, 57.75, 58.8, 59.85, 60.9, 61.95, 63, 65.1, 67.2, 69.3, 71.4, 73.5, 75.6, 77.7, 79.8, 81.9, 84])
            d5 = np.array([1.19E-03, 1.24E-03, 1.32E-03, 1.44E-03, 1.59E-03, 1.62E-03, 1.79E-03, 1.91E-03, 2.13E-03, 2.33E-03, 2.57E-03, 2.72E-03, 2.88E-03, 3.01E-03, 3.24E-03, 3.53E-03, 3.96E-03, 4.70E-03, 6.25E-03, 1.00E-02, 1.80E-02, 2.70E-02, 3.21E-02, 3.39E-02, 3.46E-02, 3.53E-02, 3.54E-02, 3.58E-02, 3.60E-02, 3.63E-02, 3.62E-02, 3.63E-02, 3.63E-02, 3.66E-02, 3.67E-02, 3.69E-02, 3.69E-02, 3.68E-02, 3.69E-02, 3.69E-02, 3.68E-02, 3.73E-02, 3.71E-02, 3.72E-02, 3.70E-02, 3.69E-02, 3.69E-02, 3.70E-02, 3.68E-02, 3.69E-02, 3.67E-02, 3.68E-02, 3.66E-02, 3.67E-02, 3.68E-02, 3.68E-02, 3.70E-02, 3.67E-02, 3.69E-02, 3.67E-02, 3.69E-02, 3.69E-02, 3.68E-02, 3.68E-02, 3.68E-02, 3.66E-02, 3.67E-02, 3.63E-02, 3.62E-02, 3.63E-02, 3.60E-02, 3.58E-02, 3.56E-02, 3.59E-02, 3.53E-02, 3.50E-02, 3.44E-02, 3.38E-02, 3.20E-02, 2.75E-02, 1.89E-02, 1.05E-02, 6.44E-03, 4.73E-03, 3.98E-03, 3.56E-03, 3.24E-03, 3.00E-03, 2.83E-03, 2.69E-03, 2.58E-03, 2.33E-03, 2.14E-03, 1.92E-03, 1.81E-03, 1.63E-03, 1.53E-03, 1.40E-03, 1.35E-03, 1.23E-03, 1.20E-03])
            d5 = d5 / d5[50] * 100
            ax4.plot(x5, d5, markersize=2, label='5 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 30],
                self.dose_grid_dose[:, 20, 30] / self.dose_grid_dose[20, 20, 30] * 100,
                # (self.dose_grid_dose[:, 20, 30] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='10 cm calc'
            )
            x10 = 0.1 * np.array([-88, -85.8, -83.6, -81.4, -79.2, -77, -74.8, -72.6, -70.4, -68.2, -66, -64.9, -63.8, -62.7, -61.6, -60.5, -59.4, -58.3, -57.2, -56.1, -55, -53.9, -52.8, -51.7, -50.6, -49.5, -48.4, -47.3, -46.2, -45.1, -44, -41.8, -39.6, -37.4, -35.2, -33, -30.8, -28.6, -26.4, -24.2, -22, -19.8, -17.6, -15.4, -13.2, -11, -8.8, -6.6, -4.4, -2.2, 0, 2.2, 4.4, 6.6, 8.8, 11, 13.2, 15.4, 17.6, 19.8, 22, 24.2, 26.4, 28.6, 30.8, 33, 35.2, 37.4, 39.6, 41.8, 44, 45.1, 46.2, 47.3, 48.4, 49.5, 50.6, 51.7, 52.8, 53.9, 55, 56.1, 57.2, 58.3, 59.4, 60.5, 61.6, 62.7, 63.8, 64.9, 66, 68.2, 70.4, 72.6, 74.8, 77, 79.2, 81.4, 83.6, 85.8, 88])
            d10 = np.array([1.39E-03, 1.49E-03, 1.58E-03, 1.64E-03, 1.75E-03, 1.90E-03, 2.04E-03, 2.16E-03, 2.35E-03, 2.52E-03, 2.80E-03, 2.83E-03, 3.08E-03, 3.16E-03, 3.38E-03, 3.61E-03, 3.87E-03, 4.36E-03, 5.34E-03, 7.59E-03, 1.27E-02, 1.93E-02, 2.35E-02, 2.52E-02, 2.61E-02, 2.65E-02, 2.66E-02, 2.69E-02, 2.70E-02, 2.72E-02, 2.72E-02, 2.74E-02, 2.76E-02, 2.79E-02, 2.79E-02, 2.80E-02, 2.82E-02, 2.83E-02, 2.84E-02, 2.85E-02, 2.85E-02, 2.85E-02, 2.85E-02, 2.84E-02, 2.85E-02, 2.84E-02, 2.84E-02, 2.84E-02, 2.85E-02, 2.83E-02, 2.83E-02, 2.86E-02, 2.84E-02, 2.85E-02, 2.84E-02, 2.83E-02, 2.84E-02, 2.84E-02, 2.82E-02, 2.84E-02, 2.83E-02, 2.83E-02, 2.82E-02, 2.80E-02, 2.80E-02, 2.79E-02, 2.79E-02, 2.77E-02, 2.75E-02, 2.74E-02, 2.72E-02, 2.70E-02, 2.71E-02, 2.67E-02, 2.67E-02, 2.65E-02, 2.59E-02, 2.56E-02, 2.44E-02, 2.16E-02, 1.60E-02, 9.74E-03, 6.17E-03, 4.72E-03, 4.08E-03, 3.67E-03, 3.52E-03, 3.22E-03, 3.03E-03, 2.93E-03, 2.77E-03, 2.63E-03, 2.42E-03, 2.22E-03, 2.07E-03, 1.90E-03, 1.82E-03, 1.69E-03, 1.54E-03, 1.49E-03, 1.39E-03])
            d10 = d10 / d10[50] * 100
            ax4.plot(x10, d10, markersize=2, label='10 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 20],
                self.dose_grid_dose[:, 20, 20] / self.dose_grid_dose[20, 20, 20] * 100,
                # (self.dose_grid_dose[:, 20, 10] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='20 cm calc'
            )
            x20 = 0.1 * np.array([-96, -93.6, -91.2, -88.8, -86.4, -84, -81.6, -79.2, -76.8, -74.4, -72, -70.8, -69.6, -68.4, -67.2, -66, -64.8, -63.6, -62.4, -61.2, -60, -58.8, -57.6, -56.4, -55.2, -54, -52.8, -51.6, -50.4, -49.2, -48, -45.6, -43.2, -40.8, -38.4, -36, -33.6, -31.2, -28.8, -26.4, -24, -21.6, -19.2, -16.8, -14.4, -12, -9.6, -7.2, -4.8, -2.4, 0, 2.4, 4.8, 7.2, 9.6, 12, 14.4, 16.8, 19.2, 21.6, 24, 26.4, 28.8, 31.2, 33.6, 36, 38.4, 40.8, 43.2, 45.6, 48, 49.2, 50.4, 51.6, 52.8, 54, 55.2, 56.4, 57.6, 58.8, 60, 61.2, 62.4, 63.6, 64.8, 66, 67.2, 68.4, 69.6, 70.8, 72, 74.4, 76.8, 79.2, 81.6, 84, 86.4, 88.8, 91.2, 93.6, 96])
            d20 = np.array([1.24E-03, 1.31E-03, 1.35E-03, 1.43E-03, 1.56E-03, 1.63E-03, 1.72E-03, 1.84E-03, 1.98E-03, 2.05E-03, 2.18E-03, 2.26E-03, 2.42E-03, 2.51E-03, 2.59E-03, 2.73E-03, 2.96E-03, 3.11E-03, 3.66E-03, 4.88E-03, 7.39E-03, 1.08E-02, 1.31E-02, 1.41E-02, 1.46E-02, 1.49E-02, 1.50E-02, 1.51E-02, 1.53E-02, 1.53E-02, 1.53E-02, 1.54E-02, 1.56E-02, 1.57E-02, 1.58E-02, 1.58E-02, 1.61E-02, 1.61E-02, 1.62E-02, 1.62E-02, 1.63E-02, 1.63E-02, 1.63E-02, 1.63E-02, 1.63E-02, 1.62E-02, 1.64E-02, 1.63E-02, 1.63E-02, 1.63E-02, 1.63E-02, 1.62E-02, 1.64E-02, 1.63E-02, 1.63E-02, 1.62E-02, 1.63E-02, 1.63E-02, 1.62E-02, 1.62E-02, 1.60E-02, 1.62E-02, 1.60E-02, 1.60E-02, 1.59E-02, 1.58E-02, 1.58E-02, 1.57E-02, 1.54E-02, 1.54E-02, 1.51E-02, 1.52E-02, 1.52E-02, 1.51E-02, 1.51E-02, 1.48E-02, 1.46E-02, 1.44E-02, 1.37E-02, 1.22E-02, 9.26E-03, 6.02E-03, 4.15E-03, 3.36E-03, 2.96E-03, 2.74E-03, 2.63E-03, 2.54E-03, 2.40E-03, 2.34E-03, 2.21E-03, 2.12E-03, 1.96E-03, 1.86E-03, 1.75E-03, 1.64E-03, 1.56E-03, 1.47E-03, 1.39E-03, 1.30E-03, 1.27E-03])
            d20 = d20 / d20[50] * 100
            ax4.plot(x20, d20, markersize=2, label='20 cm tank')

            ax4.plot(
                self.dose_grid_positions[0, :, 20, 10],
                self.dose_grid_dose[:, 20, 10] / self.dose_grid_dose[20, 20, 10] * 100,
                # (self.dose_grid_dose[:, 20, 10] /
                #  np.max(self.dose_grid_dose[:, 20, 38]) * 100),
                label='30 cm calc'
            )
            x30 = 0.1 * np.array([-104, -101.4, -98.8, -96.2, -93.6, -91, -88.4, -85.8, -83.2, -80.6, -78, -76.7, -75.4, -74.1, -72.8, -71.5, -70.2, -68.9, -67.6, -66.3, -65, -63.7, -62.4, -61.1, -59.8, -58.5, -57.2, -55.9, -54.6, -53.3, -52, -49.4, -46.8, -44.2, -41.6, -39, -36.4, -33.8, -31.2, -28.6, -26, -23.4, -20.8, -18.2, -15.6, -13, -10.4, -7.8, -5.2, -2.6, 0, 2.6, 5.2, 7.8, 10.4, 13, 15.6, 18.2, 20.8, 23.4, 26, 28.6, 31.2, 33.8, 36.4, 39, 41.6, 44.2, 46.8, 49.4, 52, 53.3, 54.6, 55.9, 57.2, 58.5, 59.8, 61.1, 62.4, 63.7, 65, 66.3, 67.6, 68.9, 70.2, 71.5, 72.8, 74.1, 75.4, 76.7, 78, 80.6, 83.2, 85.8, 88.4, 91, 93.6, 96.2, 98.8, 101.4, 104])
            d30 = np.array([9.06E-04, 9.08E-04, 9.47E-04, 1.01E-03, 1.07E-03, 1.15E-03, 1.18E-03, 1.26E-03, 1.35E-03, 1.41E-03, 1.51E-03, 1.48E-03, 1.60E-03, 1.60E-03, 1.68E-03, 1.73E-03, 1.83E-03, 1.93E-03, 2.19E-03, 2.62E-03, 3.73E-03, 5.53E-03, 7.00E-03, 7.89E-03, 8.20E-03, 8.30E-03, 8.48E-03, 8.56E-03, 8.72E-03, 8.66E-03, 8.70E-03, 8.78E-03, 8.98E-03, 8.92E-03, 8.97E-03, 9.08E-03, 9.17E-03, 9.16E-03, 9.20E-03, 9.18E-03, 9.29E-03, 9.31E-03, 9.32E-03, 9.41E-03, 9.36E-03, 9.33E-03, 9.33E-03, 9.35E-03, 9.41E-03, 9.35E-03, 9.39E-03, 9.32E-03, 9.38E-03, 9.31E-03, 9.42E-03, 9.32E-03, 9.36E-03, 9.33E-03, 9.25E-03, 9.32E-03, 9.26E-03, 9.24E-03, 9.22E-03, 9.28E-03, 9.16E-03, 9.15E-03, 9.07E-03, 9.00E-03, 8.94E-03, 8.74E-03, 8.76E-03, 8.65E-03, 8.61E-03, 8.58E-03, 8.57E-03, 8.42E-03, 8.26E-03, 8.17E-03, 7.94E-03, 7.23E-03, 5.90E-03, 4.12E-03, 2.86E-03, 2.25E-03, 1.99E-03, 1.84E-03, 1.78E-03, 1.68E-03, 1.65E-03, 1.54E-03, 1.57E-03, 1.44E-03, 1.42E-03, 1.31E-03, 1.21E-03, 1.15E-03, 1.12E-03, 1.06E-03, 1.03E-03, 9.64E-04, 9.21E-04])
            d30 = d30 / d30[50] * 100
            ax4.plot(x30, d30, markersize=2, label='30 cm tank')

            ax4.set_xlim([-25, 25])
            ax4.set_ylim([0, 110])
            ax4.set_title("Profiles")
            ax4.set_xlabel("Position [cm]")
            ax4.set_ylabel("Relative Value [%]")
            # x0, x1 = ax4.get_xlim()
            # y0, y1 = ax4.get_ylim()
            # ax4.set_aspect(abs(x1-x0)/abs(y1-y0))
            ax4.legend()

        def plot_cax():  # pylint: disable=W0612
            f5 = plt.figure()  # pylint: disable=W0612
            ax5 = plt.gca()
            ax5.plot(
                -self.dose_grid_positions[2, 20, 20, :] - 100,
                (self.dose_grid_fluence[20, 20, :] /
                np.max(self.dose_grid_fluence[20, 20, :]) * 100),
                label='Fluence'
            )
            ax5.plot(
                -self.dose_grid_positions[2, 20, 20, :] - 100,
                (self.dose_grid_terma[20, 20, :] /
                np.max(self.dose_grid_terma[20, 20, :]) * 100),
                label='TERMA'
            )
            ax5.plot(
                -self.dose_grid_positions[2, 20, 20, :] - 100,
                (self.dose_grid_dose[20, 20, :] /
                np.max(self.dose_grid_dose[20, 20, :]) * 100),
                label='Dose'
            )
            ax5.set_xlim([0, 40.0])
            ax5.set_ylim([0, 100])
            ax5.set_title("PDD")
            ax5.set_xlabel("Depth [cm]")
            ax5.set_ylabel("Relative Value [%]")
            # x0, x1 = ax5.get_xlim()
            # y0, y1 = ax5.get_ylim()
            # ax5.set_aspect(abs(x1-x0)/abs(y1-y0))
            ax5.legend()

        def plot_10x10_pdd():
            f5 = plt.figure()  # pylint: disable=W0612
            ax5 = plt.gca()
            ax5.plot(
                -self.dose_grid_positions[2, 20, 20, :] - 90,
                (self.dose_grid_dose[20, 20, :] /
                np.max(self.dose_grid_dose[20, 20, :]) * 100),
                label='Calc'
            )

            xpdd = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300])
            xpdd *= 0.1
            dpdd = np.array([1.65E-01, 1.67E-01, 1.68E-01, 1.71E-01, 1.78E-01, 1.94E-01, 2.03E-01, 2.16E-01, 2.29E-01, 2.35E-01, 2.44E-01, 2.50E-01, 2.55E-01, 2.60E-01, 2.63E-01, 2.66E-01, 2.70E-01, 2.71E-01, 2.73E-01, 2.75E-01, 2.76E-01, 2.77E-01, 2.78E-01, 2.79E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.80E-01, 2.79E-01, 2.79E-01, 2.78E-01, 2.78E-01, 2.77E-01, 2.77E-01, 2.77E-01, 2.76E-01, 2.76E-01, 2.72E-01, 2.69E-01, 2.66E-01, 2.63E-01, 2.60E-01, 2.57E-01, 2.54E-01, 2.51E-01, 2.48E-01, 2.44E-01, 2.41E-01, 2.38E-01, 2.35E-01, 2.32E-01, 2.29E-01, 2.26E-01, 2.20E-01, 2.14E-01, 2.08E-01, 2.02E-01, 1.97E-01, 1.91E-01, 1.86E-01, 1.81E-01, 1.76E-01, 1.71E-01, 1.66E-01, 1.61E-01, 1.57E-01, 1.52E-01, 1.48E-01, 1.44E-01, 1.40E-01, 1.36E-01, 1.32E-01, 1.28E-01, 1.24E-01, 1.21E-01, 1.17E-01, 1.14E-01, 1.11E-01, 1.08E-01, 1.05E-01, 1.02E-01, 9.87E-02, 9.58E-02, 9.31E-02, 9.06E-02, 8.79E-02, 8.54E-02, 8.28E-02, 8.07E-02, 7.84E-02, 7.62E-02, 7.42E-02, 7.20E-02, 7.01E-02, 6.81E-02, 6.61E-02, 6.41E-02, 6.24E-02, 6.06E-02, 5.90E-02, 5.73E-02])
            dpdd = dpdd / np.max(dpdd) * 100
            ax5.plot(xpdd, dpdd, markersize=2, label='Tank')

            ax5.set_xlim([0, 30.0])
            ax5.set_ylim([0, 110])
            ax5.set_title("PDD")
            ax5.set_xlabel("Depth [cm]")
            ax5.set_ylabel("Relative Value [%]")
            # x0, x1 = ax5.get_xlim()
            # y0, y1 = ax5.get_ylim()
            # ax5.set_aspect(abs(x1-x0)/abs(y1-y0))
            ax5.legend()

        plot_fluence()
        # plot_terma()
        plot_dose()
        # plot_10x10_profiles()
        # plot_10x10_pdd()
        plot_30x30_profiles()
        # plot_cax()
        plt.show()