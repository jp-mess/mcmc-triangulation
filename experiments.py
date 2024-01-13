
def plot_images(images):
    import matplotlib.pyplot as plt

    # Ensure there are exactly 6 images
    if len(images) != 6:
        raise ValueError("Provide exactly 6 images")
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    
    # Adjust spacing to minimize white space
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Loop through the images and display each one in the corresponding subplot
    for i, ax in enumerate(axes.flatten()):
        # Display image
        ax.imshow(images[i])
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("cars.png", bbox_inches='tight', pad_inches=0.1)
    
    plt.show()



def bmw_ring_experiment(root_dir, input_cloud=None, output_cloud_dir=None, visualize_frustums=True):
  import geometry_utils
  import image_utils
  import general_utils
  import numpy as np
  import open3d as o3d
  import matplotlib.pyplot as plt
  import os
   
  n_cameras = 6
  
  img_dim = 600
  cx = img_dim // 2
  cy = img_dim // 2
  foc = 525
  camera_parameters = general_utils.create_pinhole(foc,foc,cx,cy)

  n_points = 500000 // 30
  camera_radius = 5


  # bundle adjustment noise parameters
  pix_noise = 0.0
  pos_noise = 0.5
  rot_noise = 0.0

  ring_noise = 2


  # to find the center of the pcd, use CloudCompare or something
  center = np.array([-9.62, -0.608, -1.345])
  pcd = o3d.io.read_point_cloud(input_cloud)

  # point N cameras at the center of the point cloud (all around the same point on the sphere,
  # with slight variation)  
  up_direction = "y"
  cameras = list()

  ring_params_file = "geometry_encodings/ring_params_true.txt"
  cameras = geometry_utils.make_cameras_on_ring(center, camera_radius, up_direction, n_cameras, ring_params_file=ring_params_file)
  cameras = [general_utils.package_camera(camera, camera_parameters, 'camera_' + str(i)) for i, camera in enumerate(cameras)]
 

  # subsample point cloud indices
  indices = np.arange(len(pcd.points))
  indices = np.random.permutation(indices)
  indices = indices[:n_points]

  # correspondences[point_idx] = [(x,y), z] where (x,y) is a pixel coord and z is depth
  correspondences = image_utils.rasterize(cameras=cameras,indices_to_project=indices,
                                                          points=np.array(pcd.points),
                                                          colors=pcd.colors)


  ba_encoding_file = "problem_encodings/ring_problem_input_true.txt"
  geometry_utils.create_bal_problem_file(correspondences, n_cameras, np.array(pcd.points), cameras, ba_encoding_file,
                                         pixel_noise_scale=0.0, translation_noise_scale=0.0, rotation_noise_scale=0.0)
  
 
  # saves the simulated cameras and world points for bundle adjustment (you probably want this) 
  ba_encoding_file = "problem_encodings/ring_problem_input_noised.txt"
  geometry_utils.create_bal_problem_file(correspondences, n_cameras, np.array(pcd.points), cameras, ba_encoding_file,
                                         pixel_noise_scale=pix_noise, translation_noise_scale=pos_noise, rotation_noise_scale=rot_noise)
  
  import ring_utils 
  # projects the cameras onto an existing ring, for testing purposes (if you need an exactly correct ring problem)
  # ring_encoding_file = "problem_encodings/debug_ring_projections.txt"
  # ring_utils.update_bal_problem_with_ring_cameras(ring_params_file, ba_encoding_file, ring_encoding_file)

  # saves a noisy version of the ring parameters, for simulation testing 
  file_name, file_ext = os.path.splitext(ring_params_file)
  noisy_ring_params_file = "geometry_encodings/ring_params_noised.txt"
  ring_utils.add_noise_to_ring(ring_params_file, ring_noise, noisy_ring_params_file)
 
  if visualize_frustums: 
    import frustum_visualizer
    ring_params = ring_utils.load_ring_params(noisy_ring_params_file)
    visualizer = frustum_visualizer.PointCloudCameraVisualizer(pcd, cameras, center,ring_dict=ring_params)
    visualizer.visualize()


def bmw_retriangulation_experiment(root_dir,input_cloud=None,output_cloud_dir=None,plot_rendered_images=True, visualize_frustrums=False):
  import geometry_utils
  import image_utils
  import general_utils
  import numpy as np
  import open3d as o3d
  import matplotlib.pyplot as plt

  n_cameras = 6
  
  img_dim = 600
  cx = img_dim // 2
  cy = img_dim // 2
  foc = 525
  camera_parameters = general_utils.create_pinhole(foc,foc,cx,cy)

  n_points = 500000 // 5
  camera_radius = 5

  # to find the center of the pcd, use CloudCompare or something
  center = np.array([-9.62, -0.608, -1.345])
  pcd = o3d.io.read_point_cloud(input_cloud)

  # point N cameras at the center of the point cloud (all around the same point on the sphere,
  # with slight variation)  
  up_direction = "y"
  cameras = list()
  ring_params_file = "geometry_encodings/ring_params.txt"
  cameras = geometry_utils.make_cameras_on_ring(center, camera_radius, up_direction, n_cameras, ring_params_file=ring_params_file)
  cameras = [general_utils.package_camera(camera, camera_parameters, 'camera_' + str(i)) for i, camera in enumerate(cameras)]

  # subsample point cloud indices
  indices = np.arange(len(pcd.points))
  indices = np.random.permutation(indices)
  indices = indices[:n_points]

  # correspondences[point_idx] = [(x,y), z] where (x,y) is a pixel coord and z is depth
  correspondences = image_utils.rasterize(cameras=cameras,indices_to_project=indices,
                                                          points=np.array(pcd.points),
                                                          colors=pcd.colors)
  
  images = image_utils.render_all_images(correspondences, np.array(pcd.points), pcd.colors, img_dim)

  geometry_utils.retriangulate(cameras, correspondences, np.array(pcd.points), noise_scale=0.0, pairwise=False, save_dir=output_cloud_dir)

  if visualize_frustrums:
    import frustum_visualizer
    visualizer = frustum_visualizer.PointCloudCameraVisualizer(pcd, cameras, center)
    visualizer.visualize()

  if plot_rendered_images: 
    if len(images) == 6:
       plot_images(images)
    else:
      for image in images:
        plt.imshow(image)
        plt.show()

  return camera_parameters, images

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cov_ellipses(covariances, nstd=2, ax=None):
    """
    Plots multiple 3D covariance ellipses with dummy means for comparison.
    
    :param covariances: List of 3x3 covariance matrices.
    :param nstd: Radius of the ellipses in terms of number of standard deviations.
                 Default is 2 standard deviations.
    :param ax: Existing 3D axis. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Calculate max radii of ellipsoids to determine spacing
    max_radii = [nstd * np.sqrt(np.max(np.linalg.eigvals(cov))) for cov in covariances]
    max_dimension = np.max(max_radii)
    separation_distance = 2 * max_dimension

    # Define dummy means
    positions = [np.array([i * separation_distance, 0, 0]) for i in range(len(covariances))]

    # Colors for each ellipsoid
    colors = ['r', 'b']

    for cov, pos, color in zip(covariances, positions, colors):
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
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + pos

        # Plot
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=0.5)

    return ax



def bmw_metropolis_experiment(root_dir,input_cloud=None,output_cloud_dir=None,plot_rendered_images=False, visualize_frustrums=False):
  import geometry_utils
  import image_utils
  import general_utils
  import numpy as np
  import open3d as o3d
  import matplotlib.pyplot as plt

  n_cameras = 3
  
  img_dim = 600
  cx = img_dim // 2
  cy = img_dim // 2
  foc = 525
  camera_parameters = general_utils.create_pinhole(foc,foc,cx,cy)

  n_points = 500000 // 30
  camera_radius = 5

  angle_distance_small = 5
  angle_distance_large = 100

  # to find the center of the pcd, use CloudCompare or something
  center = np.array([-9.62, -0.608, -1.345])
  pcd = o3d.io.read_point_cloud(input_cloud)

  # point N cameras at the center of the point cloud (all around the same point on the sphere,
  # with slight variation)  
  up_direction = "y"
  # subsample point cloud indices
  indices = np.arange(len(pcd.points))
  indices = np.random.permutation(indices)
  indices = indices[:n_points]


  cameras = list()
  cameras = geometry_utils.make_cameras(center, camera_radius, up_direction, n_cameras, distance=angle_distance_small)
  cameras = [general_utils.package_camera(camera, camera_parameters, 'camera_' + str(i)) for i, camera in enumerate(cameras)]

  # correspondences[point_idx] = [(x,y), z] where (x,y) is a pixel coord and z is depth
  correspondences = image_utils.rasterize(cameras=cameras,indices_to_project=indices,
                                                          points=np.array(pcd.points),
                                                          colors=pcd.colors)
  output_file = "small_angles.txt"
  geometry_utils.create_bal_problem_file(correspondences, n_cameras, np.array(pcd.points), cameras, output_file, translation_noise_scale = 0.0, rotation_noise_scale = 0.0, pixel_noise_scale = 0.0)

  cameras = list()
  cameras = geometry_utils.make_cameras(center, camera_radius, up_direction, n_cameras, distance=angle_distance_large)
  cameras = [general_utils.package_camera(camera, camera_parameters, 'camera_' + str(i)) for i, camera in enumerate(cameras)]
  # correspondences[point_idx] = [(x,y), z] where (x,y) is a pixel coord and z is depth
  correspondences = image_utils.rasterize(cameras=cameras,indices_to_project=indices,
                                                          points=np.array(pcd.points),
                                                          colors=pcd.colors)
  output_file = "large_angles.txt"
  geometry_utils.create_bal_problem_file(correspondences, n_cameras, np.array(pcd.points), cameras, output_file, translation_noise_scale = 0.0, rotation_noise_scale = 0.0, pixel_noise_scale = 0.0)




