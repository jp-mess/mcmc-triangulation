

import numpy as np
from scipy.spatial.transform import Rotation as R

def project_point_debug():
    
    camera_vector = np.array([-0.02313, -0.0078111, 3.0754, -10.3086, 1.55502, 3.11002, 525, 300, 300])
    world_point = np.array([-9.34843, -0.245021, -3.3983]) 

    rotation_vector = camera_vector[:3]
    translation = camera_vector[3:6]
    focal = camera_vector[6]
    cx = camera_vector[7]
    cy = camera_vector[8]
    intrinsic_params = np.array([[focal,0,cx],[0,focal,cy],[0,0,1]])

    print("rotation",rotation_vector)
    print("translation",translation)
    print("intrinsic",intrinsic_params)    
    print("world", world_point)

    # Convert rotation vector (angle-axis) to rotation matrix
    rot_mat = R.from_rotvec(rotation_vector).as_matrix()

    # Apply extrinsic parameters (rotation and translation)
    #transformed_point = rot_mat @ world_point + translation

    pose_mat = np.eye(4)
    pose_mat[:3, :3] = rot_mat
    pose_mat[:3, 3] = translation

    # Convert world point to homogeneous coordinates
    world_point_homogeneous = np.append(world_point, 1)

    # Apply extrinsic parameters (pose matrix)
    transformed_point_homogeneous = np.linalg.pinv(pose_mat) @ world_point_homogeneous
    #transformed_point_homogeneous = pose_mat @ world_point_homogeneous
    
    point_hom = transformed_point_homogeneous[:3]  
      
    pixel = np.matmul(intrinsic_params,point_hom)
    pixel = pixel[:2] / pixel[2]
    pixel = pixel.flatten().astype(float)

    print(pixel)


    #xp = intrinsic[0, 0] * transformed_point_homogeneous[0] + intrinsic[0, 2] * transformed_point_homogeneous[2]
    #yp = intrinsic[1, 1] * transformed_point_homogeneous[1] + intrinsic[1, 2] * transformed_point_homogeneous[2]
    #u = xp / transformed_point_homogeneous[2]
    #v = yp / transformed_point_homogeneous[2]

    #return np.array([u, v])


"""
wrapper function for getting a list of images from a correspondence 
object

correspondences[point_idx] = [ [(x,y), depth], [(x,y), depth]... ]

where each point_idx is the index of a unique point in the original point cloud

the list [(x,y), depth] is the projected image coordinate (x,y) and depth
for one of the images that views that point (this simulation is kind of trivial
so all images will view all points, ignoring occlusion) 

"""
def render_all_images(correspondences, points, colors, img_dim):
  images = dict()
  color_map = dict()
  for point_idx in correspondences.keys():
    for idx,pixels in enumerate(correspondences[point_idx]):
      if idx not in images:
        images[idx] = list()
        color_map[idx] = list()
      images[idx].append(pixels)
      color_map[idx].append(colors[point_idx])
  rendered = list()
  for idx in images.keys():
    rendered.append(render_coordinates(images[idx], (img_dim, img_dim), color_map[idx]))
  return rendered

def rectilinear_distort(point, k, p):
    """
    Apply rectilinear (radial and tangential) distortion to a 2D point.
    point: A 2D point in the normalized image plane (x, y).
    k: List or array of radial distortion coefficients (at least two: k1, k2).
    p: List or array of tangential distortion coefficients (p1, p2).
    """
    y, x = point
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    xy = x * y

    # Radial distortion
    k1, k2 = k[0], k[1]
    radial_distortion = (1 + k1 * r2 + k2 * r2 * r2)

    x_radial = x * radial_distortion
    y_radial = y * radial_distortion

    # Tangential distortion
    p1, p2 = p[0], p[1]
    x_tangential = 2 * p1 * xy + p2 * (r2 + 2 * x2)
    y_tangential = p1 * (r2 + 2 * y2) + 2 * p2 * xy

    # Apply distortion
    x_distorted = x_radial + x_tangential
    y_distorted = y_radial + y_tangential

    return np.array([y_distorted, x_distorted])


def fisheye_distort(pixel_point, k, intrinsic_params):
    """
    Apply fisheye distortion to a 2D point in pixel coordinates.
    pixel_point: A 2D point in pixel coordinates (x, y).
    k: Distortion coefficient.
    intrinsic_params: used to normalize image coordinates
    """
    fx = fy = intrinsic_params[0,0]
    cx = intrinsic_params[0,2]
    cy = intrinsic_params[1,2]

    # Convert from pixel coordinates to normalized image coordinates
    x = (pixel_point[1] - cx) / fx
    y = (pixel_point[0] - cy) / fy

    r = np.sqrt(x**2 + y**2)  # Radial distance from the center
    theta = np.arctan(r)

    # Apply fisheye distortion
    theta_d = theta * (1 + k * theta**2)
  
    # Scale back to normalized image plane
    x_distorted = (theta_d / r) * x if r != 0 else x
    y_distorted = (theta_d / r) * y if r != 0 else y

    # Optionally, convert back to pixel coordinates
    x_pixel_distorted = x_distorted * fx + cx
    y_pixel_distorted = y_distorted * fy + cy

    return np.array([y_pixel_distorted, x_pixel_distorted])




def rasterize(cameras, indices_to_project, points, colors):
    import numpy as np
    import copy

    cameras = copy.deepcopy(cameras)

    pixels = list()
    correspondences = dict()

    for camera_obj in cameras:
        # Retrieve the fisheye distortion coefficient
        k = camera_obj['distortion']

        for point_idx in indices_to_project:
            pose = camera_obj['extrinsic']
            intrinsic_params = camera_obj['intrinsic']
            camera_center = pose[:3, 3]

            point_hom = np.append(points[point_idx, :], 1).reshape((4, 1))
            point_hom = np.matmul(np.linalg.pinv(pose), point_hom)
            point_hom = point_hom[:3]

            # Compute pixel coordinates before distortion
            pixel = np.matmul(intrinsic_params, point_hom)
            pixel = pixel[:2] / pixel[2]
            pixel = pixel.flatten().astype(float)

            # Apply fisheye distortion
            pixel_distorted = fisheye_distort(pixel, k, intrinsic_params)
            #print(pixel, pixel_distorted)
            pixel = pixel_distorted

            pixels.append(pixel)

            if point_idx not in correspondences:
                correspondences[point_idx] = list()

            correspondences[point_idx].append((pixel, np.linalg.norm(point_hom)))

    return correspondences



"""
coordinates is a dictionary with both pixel information,
as well as depth (for basic rendering)

coordinates[point_idx] = [ (x,y), z ]

image_size is a tuple (dimensions)

colors is a list where colors[point_idx] = [r,g,b]

"""
def render_coordinates(coordinates, image_size, colors, downsample_factor=2):
  import numpy as np
  import cv2

  image = np.ones((image_size[0], image_size[1], 3))
  collisions = dict()
  for i, coord in enumerate(coordinates):
    x, y = coord[0]
    x = int(round(x))
    y = int(round(y))

    # Flip the x-coordinate
    x = image_size[1] - 1 - x

    # Flip the y-coordinate
    y = image_size[0] - 1 - y

    # color in if not viewed
    if (x, y) not in collisions:
      collisions[(x, y)] = coord[1]
      if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
        image[y, x] = colors[i]
    else:
      if collisions[(x, y)] > coord[1]:
        collisions[(x, y)] = coord[1]
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
          image[y, x] = colors[i]

  # Downsample the image to make it look lower resolution
  downsampled_image = cv2.resize(image, (image_size[1] // downsample_factor, image_size[0] // downsample_factor),
                                   interpolation=cv2.INTER_LINEAR)

  # Upsample it back to original resolution to enlarge the points
  upsampled_image = cv2.resize(downsampled_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

  return upsampled_image


if __name__ == "__main__":
  project_point_debug()

