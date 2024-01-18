import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import numpy as np

def load_bal_problem_file(bal_file):
    with open(bal_file, 'r') as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        # Skip the observation data
        for _ in range(n_observations):
            file.readline()

        # Read the camera parameters
        cameras = []
        for _ in range(n_cameras):
            angle_axis = np.array([float(file.readline()) for _ in range(3)])
            translation = np.array([float(file.readline()) for _ in range(3)])
            intrinsic = np.array([float(file.readline()) for _ in range(3)])
            rotation_matrix = R.from_rotvec(angle_axis).as_matrix()

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation_matrix
            extrinsic[:3, 3] = translation

            camera = {'extrinsic': extrinsic, 'intrinsic': np.eye(3)}
            camera['intrinsic'][0, 0] = intrinsic[0]  # Focal length
            camera['intrinsic'][0, 2] = intrinsic[1]  # cx
            camera['intrinsic'][1, 2] = intrinsic[2]  # cy

            cameras.append(camera)

        # Read the 3D points
        points = np.array([[float(file.readline()) for _ in range(3)] for _ in range(n_points)])

    return cameras, points

def invert_camera_pose(bal_file, output_file):
    with open(bal_file, 'r') as file:
        lines = file.readlines()

    header = lines[0]
    n_cameras, n_points, n_observations = map(int, header.split())

    camera_data_start_index = 1 + n_observations
    camera_data_end_index = camera_data_start_index + n_cameras * 9

    with open(output_file, 'w') as file:
        file.write(header)

        # Write the observations unchanged
        for line in lines[1:camera_data_start_index]:
            file.write(line)

        # Process and write inverted camera parameters
        for i in range(n_cameras):
            camera_start = camera_data_start_index + i * 9
            camera_end = camera_start + 9
            camera_data = lines[camera_start:camera_end]

            # Extract rotation and translation
            rotation = np.array([float(camera_data[j]) for j in range(3)])
            translation = np.array([float(camera_data[j+3]) for j in range(3)])

            # Convert to pose matrix
            rotation_matrix = R.from_rotvec(rotation).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation

            # Invert the pose matrix
            inverted_matrix = np.linalg.inv(pose_matrix)

            # Extract the inverted rotation and translation
            inverted_rotation = R.from_matrix(inverted_matrix[:3, :3]).as_rotvec()
            inverted_translation = inverted_matrix[:3, 3]

            # Write inverted camera parameters
            for component in inverted_rotation:
                file.write(f'{component}\n')
            for component in inverted_translation:
                file.write(f'{component}\n')

            # Write intrinsics and any other camera data unchanged
            for j in range(6, 9):
                file.write(camera_data[j])

        # Write the rest of the file unchanged (world points)
        for line in lines[camera_data_end_index:]:
            file.write(line)


def plot_3d_points(points):
    """
    Plot a list of 3D points using matplotlib.

    :param points: A list of numpy arrays, each of shape (3,), representing 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, y, and z coordinates from the points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]

    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()


def create_bal_problem_file(correspondences, n_cameras, point_container, true_cameras, output_file, translation_noise_scale = 0.0, rotation_noise_scale = 0.0, pixel_noise_scale = 0.0):

    from scipy.spatial.transform import Rotation as R
    import sys
    import numpy as np
    import transforms3d

    n_points = len(correspondences)
    point_idx_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(correspondences.keys())}

    with open(output_file, 'w') as file:
        # Write the header
        num_observations = n_cameras * n_points
        file.write(f"{n_cameras} {n_points} {num_observations}\n")

        # Write the observations
        for original_idx, observations in correspondences.items():
            new_idx = point_idx_mapping[original_idx]
            for camera_idx in range(n_cameras):
                row, col = observations[camera_idx][0]  # Extract pixel coordinates
                noise = np.random.normal(0,pixel_noise_scale,size=2)
                row += noise[0]
                col += noise[1]
                file.write(f"{camera_idx} {new_idx} {row} {col}\n")

         # Write the true camera parameters
        for camera in true_cameras:

            extrinsic_copy = np.copy(camera["extrinsic"])
            #extrinsic_copy = np.linalg.pinv(extrinsic_copy) 
 
            # Convert the rotation matrix to Euler angles (angle-axis)
            rotation_matrix = extrinsic_copy[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            angle_axis = rotation.as_rotvec()  # Convert to angle-axis
            angle_axis += np.random.normal(0, rotation_noise_scale, 3)

            # Write the angle-axis rotation
            for i in range(3):
                file.write(f"{angle_axis[i]}\n")

            # Write the translation
            translation_vector = extrinsic_copy[:3, 3]
            translation_vector += np.random.normal(0, translation_noise_scale, 3)
            for i in range(3):
                file.write(f"{translation_vector[i]}\n")

            # Write the focal length, cx, and cy from the intrinsic matrix
            intrinsic = camera["intrinsic"]
            file.write(f"{intrinsic[0, 0]}\n")  # Focal length
            file.write(f"{intrinsic[0, 2]}\n")  # cx
            file.write(f"{intrinsic[1, 2]}\n")  # cy

        # Write the true 3D points
        for original_idx in correspondences.keys():
            point = point_container[original_idx]
            for i in range(3):
                file.write(f"{point[i]}\n")

def make_cameras_on_ring(point_cloud_center, radius, up_direction, num_cameras, ring_params_file=None):
    import numpy as np
    import general_utils
    import os

    if up_direction != "y":
        raise NotImplementedError("Currently only 'y' up-direction is implemented.")

    # Randomly choose an elevation degree between 60 and 90
    elevation_degree = np.random.uniform(60, 90)
    elevation_radian = np.radians(elevation_degree)
    phi = elevation_radian  # phi is the elevation angle from the vertical axis

    # Calculate the actual center of the ring
    ring_center = point_cloud_center + np.array([0, radius * np.cos(phi), 0])  # Assuming 'y' up-direction

    def camera_position_on_ring(angle):
        # Calculate x, y, z based on spherical coordinates
        x = radius * np.sin(phi) * np.cos(angle)
        z = radius * np.sin(phi) * np.sin(angle)
        y = radius * np.cos(phi)
        return np.array([x, y, z]) + point_cloud_center

    def camera_direction(camera_loc):
        center_proj_ray = point_cloud_center - camera_loc
        return center_proj_ray / np.linalg.norm(center_proj_ray)

    def make_pose_matrix(camera_loc):
        z_dir = camera_direction(camera_loc)  # Forward direction (Z-axis)
        x_dir = np.cross(np.array([0, 1, 0]), z_dir)  # Right direction (X-axis)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)  # Up direction (Y-axis)
        y_dir /= np.linalg.norm(y_dir)
        R_wc = np.stack((x_dir, y_dir, z_dir), axis=1)
        return general_utils.create_pose_matrix(R_wc, camera_loc)

    cameras = []
    locs = list()
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras  # Uniformly space cameras around the ring
        camera_loc = camera_position_on_ring(angle)
        pose_matrix = make_pose_matrix(camera_loc)
        cameras.append(pose_matrix)
        locs.append(pose_matrix[:3,3])

    ring_params = {
        "type": "ring",
        "center": ",".join(map(str, ring_center)),
        "normal": "0,1,0",  # Assuming 'y' up-direction
        "radius": str(radius * np.sin(phi)),
        "elevation_degree": str(elevation_degree)
    }
  
    if ring_params_file is not None:
      with open(ring_params_file, "w") as file:
          for key, value in ring_params.items():
              file.write(f"{key}: {value}\n")

    return cameras



import numpy as np
import general_utils

def make_cameras(point_cloud_center, radius, up_direction, num_cameras, distance=None):
    if up_direction != "y":
        raise NotImplementedError("Currently only 'y' up-direction is implemented.")

    def random_camera_position():
        theta = np.random.uniform(0, np.pi * 2)
        phi = np.random.uniform(np.pi / 6, np.pi / 2)  # Sampling from 30 to 90 degrees
        return spherical_to_cartesian(theta, phi), theta, phi

    def spherical_to_cartesian(theta, phi):
        x = radius * np.sin(phi) * np.cos(theta)
        z = radius * np.sin(phi) * np.sin(theta)
        y = radius * np.cos(phi)
        return np.array([x, y, z]) + point_cloud_center

    def camera_direction(camera_loc):
        center_proj_ray = point_cloud_center - camera_loc
        return center_proj_ray / np.linalg.norm(center_proj_ray)

    def make_pose_matrix(camera_loc):
        z_dir = camera_direction(camera_loc)  # Forward direction (Z-axis)
        x_dir = np.cross(np.array([0, 1, 0]), z_dir)  # Right direction (X-axis)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)  # Up direction (Y-axis)
        y_dir /= np.linalg.norm(y_dir)
        R_wc = np.stack((x_dir, y_dir, z_dir), axis=1)
        return general_utils.create_pose_matrix(R_wc, camera_loc)

    cameras = []

    # Generate the first camera
    first_camera_loc, theta, phi = random_camera_position()
    first_pose_matrix = make_pose_matrix(first_camera_loc)
    cameras.append(first_pose_matrix)

    # Generate subsequent cameras
    for _ in range(1, num_cameras):
        if distance is not None:
            # Logic to generate camera positions at a specified distance from the first camera
            # This is a simplified approach and might need refinement for precise control
            perturbation = np.random.uniform(-1*distance, distance)
            theta += (perturbation * (np.pi / 180))
            perturbation = np.random.uniform(-1*distance, distance)
            phi += (perturbation * (np.pi / 180))
            new_camera_loc = spherical_to_cartesian(theta,phi)
        else:
            new_camera_loc = random_camera_position()

        new_pose_matrix = make_pose_matrix(new_camera_loc)
        cameras.append(new_pose_matrix)

    return cameras



"""
assuming xyz is a vector centered at 0, sample a new coord
on the same sphere

up direction matters!
"""
def nearby_coord(xyz,variation=0.25,up_dir='z'):
  import numpy as np

  r = np.linalg.norm(xyz)
  theta = np.arccos(xyz[2] / r)
  phi = np.arctan2(xyz[1],xyz[0])

  theta += np.random.normal(0,variation)
  phi += np.random.normal(0,variation)
  
  x = r * np.sin(theta) * np.cos(phi)
  y = r * np.sin(theta) * np.sin(phi)
  z = r * np.cos(theta)

  return np.array([x,y,z])


"""
cameras : list of N cameras where each camera is a dictionary:

camera['camera'] = pose matrix (4x4)
camera['name'] = string identifier for the camera

pixels is a list of pixel coordinates, one for each camera

[(x1,y1), (x2,y2), ...(xN, yN)] in (row,column) format (not x,y)

pairwise will generate (N choose 2) points, multiview triangulates one 
""" 
def DLT(cameras,pixels,pairwise=True):
  import numpy as np
  import general_utils
  import itertools

  points = list()
  if pairwise:
    pairs = list(itertools.combinations(list(range(len(cameras))),2))
    
    for i,j in pairs:
      # [(x,y) z]
      point1 = pixels[i]
      point2 = pixels[j]
      camera_params = cameras[i]["intrinsic"] 

      camA = np.linalg.inv(cameras[i]["extrinsic"])
      rotA, transA = general_utils.get_rot_trans(camA)
      camB = np.linalg.inv(cameras[j]["extrinsic"])
      rotB, transB = general_utils.get_rot_trans(camB)
  
      camA = np.concatenate([rotA,transA],axis=-1)
      P1 = camera_params @ camA
      camB = np.concatenate([rotB,transB],axis=-1)
      P2 = camera_params @ camB

      A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
      A = np.array(A).reshape((4,4))
      B = A.transpose() @ A
      from scipy import linalg
      U,s,Vh = linalg.svd(B,full_matrices=False)
      point = Vh[3,0:3]/Vh[3,3]
      points.append(point)
    return points
  else:
    A = list()
    for i in range(len(cameras)):
      # [(x,y) z]
      point = pixels[i]
      camA = np.linalg.inv(cameras[i]["extrinsic"])
      camera_params = cameras[i]["intrinsic"]
      rotA, transA = general_utils.get_rot_trans(camA)
      camA = np.concatenate([rotA, transA], axis=-1)
      P = camera_params @ camA
      A_ = np.array([point[1]*P[2,:] - P[1,:],
                    P[0,:] - point[0]*P[2,:]])
      A.append(A_)
    A = np.concatenate(A,axis=0)
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
    point = Vh[3,0:3]/Vh[3,3]
    return [point] 


"""
cameras : list of N cameras where each camera is a dictionary:

camera['camera'] = pose matrix (4x4)
camera['name'] = string identifier for the camera

correspondences is a dictionary with both pixel information,
as well as depth (for basic rendering)

coordinates[point_idx] = [ (x,y), depth ]

points is an np.array list of 3D point

noise_scale is how much gaussian noise to add to each pixel observation

pairwise = True will perform pairwise triangulation (one world point for
each view edge, while pairwise = False will do multiview triangulation
i.e. one world point for all N cameras that saw the same point_idx)
""" 
def retriangulate(cameras, correspondences, points, noise_scale=0.0, do_lm=True, pairwise=True, save_dir=None):
  import numpy as np
  import open3d as o3d
  import os

  if len(cameras) < 2:
    print("not enough cameras to perform triangulation")
    return None

  triangulated = list()
  errors = list()

  from tqdm import tqdm

  for point_idx in tqdm(correspondences):
    # each coord is [(x,y) z] and we just want the (x,y) pixels
    pixels = [coords[0] for coords in correspondences[point_idx]]
    for pix_idx in range(len(pixels)):
      pixels[pix_idx] = pixels[pix_idx] + np.random.normal(0,noise_scale,size=2)
      
      if do_lm:
        result = nonlinear_triangulation(cameras, pixels, pairwise=pairwise)
      else:
        result = DLT(cameras, pixels, pairwise=pairwise)

      triangulated.extend(result)
      
      err = np.mean([np.linalg.norm(p - points[point_idx,:]) for p in result])
      errors.append(err)

  triangulated = np.vstack(triangulated)
  triangulated = np.array([triangulated[i,:] for i in range(triangulated.shape[0])])
  new_pcd = o3d.geometry.PointCloud()
  new_pcd.points = o3d.utility.Vector3dVector(triangulated.squeeze())
  print(f"noise level {np.array(noise_scale)}\t\tn_cameras {len(cameras)}\t\tavg error {np.mean(errors)}\t\tmultiview? {not pairwise}")

  if save_dir is not None:
    name = "retriangulated_" + str(noise_scale) + "_" + str(len(cameras)) + "_cameras"
    if pairwise:
      name += str("_pairwise.ply")
    else:
      name += str("_multiview.ply")
    name = os.path.join(save_dir,name)
    o3d.io.write_point_cloud(name, new_pcd)

  return triangulated


def project_point_to_image(point_3d, camera):
    """
    Projects a 3D point onto the image plane of the camera.

    :param point_3d: The 3D point in world coordinates (numpy array or list of length 3).
    :param camera: The camera dictionary containing 'extrinsic' and 'intrinsic' matrices.
    :return: 2D point in image coordinates.
    """

    # Convert the point to homogeneous coordinates
    point_3d_homog = np.append(point_3d, 1)

    # Invert the extrinsic matrix to transform from world to camera coordinates
    extrinsic_inv = np.linalg.inv(camera['extrinsic'])

    # Transform the point to camera coordinates
    point_cam = np.dot(extrinsic_inv, point_3d_homog)

    # Use the intrinsic matrix to project onto the image plane
    intrinsic = camera['intrinsic']
    point_img_homog = np.dot(intrinsic, point_cam[:3])

    # Convert from homogeneous coordinates to 2D
    point_img = point_img_homog[:2] / point_img_homog[2]

    return point_img

def reproject_point(camera, point_3d):
    import image_utils
    reprojected_point = project_point_to_image(point_3d, camera)
    updated_point = image_utils.fisheye_distort(reprojected_point, camera["distortion"], camera["intrinsic"])
    return updated_point

from scipy.optimize import least_squares
import itertools

def nonlinear_triangulation(cameras, pixels, pairwise=False):
    def reprojection_error(point_3d):
        error = []
        for camera, pixel in zip(cameras, pixels):
            reprojected_pixel = reproject_point(camera, point_3d)
            error.append(reprojected_pixel - pixel)
        return np.array(error).flatten()
   
    if pairwise: 
      pairs = list(itertools.combinations(list(range(len(cameras))),2))
      results = []
      for i,j in pairs:
        pixels_ = [pixels[idx] for idx in [i,j]]
        cameras_ = [cameras[idx] for idx in [i,j]]
        initial_guess = DLT(cameras_, pixels_, pairwise=True)
        results.append(least_squares(reprojection_error, initial_guess[0], method='lm').x)
      return results
    else: 
      initial_guess = DLT(cameras, pixels, pairwise=False)
      result = least_squares(reprojection_error, initial_guess[0], method='lm')
      return [result.x]



def compute_mean_reprojection_error(cameras, correspondences, point_cloud, corr_idx, camera_indices, point_idx):
    """
    Computes the mean reprojection error of a specific point.

    :param cameras: The dictionary of correspondences
    :param correspondences: The dictionary of correspondences for each point.
    :param point_cloud: The np.array of 3D points from the retriangulation.
    :param point_idx: The index of the point to analyze.
    :return: The mean reprojection error for the point.
    """

    # Extract the 3D point using point_idx
    point_3d = point_cloud[point_idx]

    # Initialize an error accumulator
    total_error = 0.0

    # Loop through each correspondence for the point
    for cam_idx, (pixel_coord, depth) in enumerate(correspondences[corr_idx]):
        if cam_idx not in camera_indices:
          continue

        # Reproject the 3D point onto the camera plane
        # Note: You'll need camera parameters and a reprojection function
        reprojected_point = project_point_to_image(point_3d, cameras[cam_idx])

        # Compute the error (e.g., Euclidean distance) between reprojected point and actual pixel coordinate
        error = np.linalg.norm(np.array(pixel_coord) - np.array(reprojected_point))

        # Accumulate the error
        total_error += error

    # Compute the mean error
    mean_error = total_error / len(correspondences[corr_idx])

    return mean_error

def update_covariance_matrix(cov_old, mean_old, x_new, n):
    """
    Update the covariance matrix with a new sample.

    :param cov_old: The old covariance matrix.
    :param mean_old: The old mean of samples.
    :param x_new: The new sample to be added.
    :param n: The number of samples used to compute the old covariance matrix.
    :return: Updated covariance matrix and updated mean.
    """
    x_new = np.array(x_new)
    mean_new = (n * mean_old + x_new) / (n + 1)
    cov_new = (n / (n + 1)) * cov_old + (n / (n + 1)**2) * np.outer(x_new - mean_old, x_new - mean_old)

    return cov_new, mean_new


def gaussian_error_function(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))



def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    """
    Plots a 3D covariance ellipse.
    
    :param cov: 3x3 covariance matrix.
    :param pos: 3D position of the center of the ellipse (mean).
    :param nstd: Radius of the ellipse in terms of number of standard deviations.
                 Default is 2 standard deviations.
    :param ax: Existing 3D axis. If None, a new figure is created.
    :param kwargs: Additional arguments for the plot.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Decompose the covariance matrix
    U, s, rotation = np.linalg.svd(cov)
    radii = nstd * np.sqrt(s)

    # Generate data for the ellipsoid
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Rotate and position the ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], rotation) + pos

    # Plot
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b', alpha=0.5, **kwargs)

    return ax



def mh_analysis_for_point(cameras, correspondences, triangulated, camera_indices, point_idx):
    """
    Performs Metropolis-Hastings analysis on a specific point from the point cloud.

    :param correspondences: The dictionary of correspondences for each point.
    :param triangulated: The np.array of 3D points from the retriangulation.
    :param point_idx: The index of the point to analyze.
    """

    corr_idx = list(correspondences.keys())[point_idx]

    # Compute the initial mean reprojection error for the point
    initial_error = compute_mean_reprojection_error(cameras, correspondences, triangulated, corr_idx, camera_indices, point_idx)

    max_error = 1

    variance = np.eye(3) * 0.001
    mean = triangulated[point_idx]


    samples = [mean]

    cams = [cameras[idx] for idx in camera_indices]
    
    pixel_coords = [project_point_to_image(mean, cam) for cam in cams]

    print("initializing MCMC")
    for _ in range(10000):
      next_point = np.random.multivariate_normal(mean, variance)
      errors = []
      for idx, cam in enumerate(cams):
        projected = project_point_to_image(next_point, cam)
        error = np.linalg.norm(np.array(pixel_coords[idx]) - np.array(projected))
        errors.append(error)
      error = np.mean(np.array(errors))

      # don't consider this one
      if error > max_error:
        continue 

      samples.append(next_point)

    variance = np.cov(np.array(samples), rowvar=False)

    print("accepted initial points", len(samples))
    print("variance", variance)

    samples = [mean]
    counter = 0
    #previous_fx = initial_error
    previous_fx = 1
    from tqdm import tqdm
    for _ in tqdm(range(100000)):
      next_point = np.random.multivariate_normal(mean, variance)
      errors = []
      for idx, cam in enumerate(cams):
        projected = project_point_to_image(next_point, cam)
        error = np.linalg.norm(np.array(pixel_coords[idx]) - np.array(projected))
        errors.append(error)
      error = np.mean(np.array(errors))

      if error > max_error:
        continue
 
      accept = np.random.uniform(0,1)
      # opposite of mcmc, want to find the tails
      #current_fx = error
      current_fx = gaussian_error_function(error, mu=initial_error, sigma=1)
      ratio = min(1, current_fx / previous_fx)
      if accept < ratio:
        samples.append(next_point)
        previous_fx = current_fx
        counter += 1
      if counter > 100:
        variance = np.cov(np.array(samples), rowvar=False)
        counter = 0

    variance = np.cov(np.array(samples), rowvar=False)
    return mean, variance








