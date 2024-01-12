import open3d as o3d
import numpy as np
from dataclasses import dataclass

@dataclass
class WhiteBackground:
    background = [1, 1, 1]
    frustum = [0, 0, 0]
    direction = [0, 1, 0]

@dataclass
class CharcoalBackground:
    background = [0.1, 0.1, 0.1]
    frustum = [1,1,1]
    direction = [0,1,0]



class PointCloudCameraVisualizer:
    def __init__(self, pcd, cameras, center_point, ring_dict=None, draw_green_directions=True):
        if isinstance(pcd, str):
            self.pcd = o3d.io.read_point_cloud(pcd)
        else:
            self.pcd = pcd

        self.cameras = cameras
        self.center_point = center_point
        self.ring_dict = ring_dict
        self.draw_green_directions = draw_green_directions
        self.cmap = WhiteBackground()

    def create_plane(self, center, normal, size=1.0, thickness=0.01, color=[0.53, 0.81, 0.92]):
        box = o3d.geometry.TriangleMesh.create_box(width=size, height=thickness, depth=size)
        # Align box with the normal vector
        rotation_matrix = self._align_vector_to_another(np.array([0, 1, 0]), normal)
        box.rotate(rotation_matrix, center=False)
        box.translate(center - np.array([0, thickness / 2, 0]))
        box.paint_uniform_color(color)  # Set color
        return box


    def create_ring(self, center, normal, outer_radius, thickness=0.05, color=[0.49, 0.98, 1.0]):
        """
        Create a ring by modifying a torus. Outer radius is the radius of the ring,
        and thickness is the radius of the torus tube.
        """
        # Ensure outer_radius is a float, not an array
        outer_radius = float(outer_radius) 

        torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=outer_radius, tube_radius=thickness, radial_resolution=30, tubular_resolution=20)
        # Align torus with the normal vector
        rotation_matrix = self._align_vector_to_another(np.array([0, 0, 1]), normal)
        torus.rotate(rotation_matrix)
        torus.translate(center)
        torus.paint_uniform_color(color)  # Set color
        return torus


    def _align_vector_to_another(self, v1, v2):
      """
      Compute the rotation matrix that aligns v1 to v2
      """
      v = np.cross(v1, v2)
      c = np.dot(v1, v2)
      s = np.linalg.norm(v)

      if s < 1e-10:  # Threshold to check if vectors are parallel
          # If vectors are nearly parallel, no rotation is needed
          return np.eye(3) if c > 0 else -np.eye(3)  # Handle opposite direction case

      kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
      rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
      return rotation_matrix


    def create_parametrized_ring(self, ring_params, color=[0.49, 0.98, 1.0]):
        center = np.array(ring_params['center'])
        normal = np.array(ring_params['normal'])
        radius = ring_params['radius']
        ring = self.create_ring(center, normal, radius)
        ring.paint_uniform_color(color)
        return ring



    def create_camera_marker(self, location, size=0.05):
        return o3d.geometry.TriangleMesh.create_sphere(radius=size).translate(location)

    def create_direction_line(self, camera_extrinsic, length=5):
        camera_position = camera_extrinsic[:3, 3]
        camera_direction = camera_extrinsic[:3, 2]  # Z-axis (forward direction) of the camera

        # Calculate the end point of the line
        end_point = camera_position + camera_direction * length

        # Create the line
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([camera_position, end_point])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color(self.cmap.direction)
        return line

    def create_wireframe_box(self, center, extents, rotation_matrix):
        corners = [
            center + np.dot(rotation_matrix, np.array([dx, dy, dz]))
            for dx in [-extents[0] / 2, extents[0] / 2]
            for dy in [-extents[1] / 2, extents[1] / 2]
            for dz in [-extents[2] / 2, extents[2] / 2]
        ]
        lines = [
            [0, 1], [0, 2], [1, 3], [2, 3],
            [4, 5], [4, 6], [5, 7], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(corners)
        wireframe.lines = o3d.utility.Vector2iVector(lines)
        wireframe.paint_uniform_color(self.cmap.frustum)
        return wireframe

    def create_frustum(self, camera_extrinsic, dim=0.2):
        # Extract camera position and rotation from the extrinsic matrix
        camera_position = camera_extrinsic[:3, 3]
        rotation_matrix = camera_extrinsic[:3, :3]

        # The extents of the frustum
        extents = np.array([dim, dim, dim * 1.25])  # Making the depth 1.25 times the width/height

        # Create the wireframe box representing the frustum
        wireframe_box = self.create_wireframe_box(camera_position, extents, rotation_matrix)
        return wireframe_box

    def visualize(self):
        visual_elements = [self.pcd]
        for camera in self.cameras:
            camera_marker = self.create_camera_marker(camera['extrinsic'][:3, 3])
            frustum = self.create_frustum(camera['extrinsic'])
            if self.draw_green_directions:
              line = self.create_direction_line(camera['extrinsic'])
              visual_elements.extend([camera_marker, line, frustum])
            else:
              visual_elements.extend([camera_marker, frustum])
            

        # Add the ring if ring_dict is provided
        if self.ring_dict:
            ring = self.create_parametrized_ring(self.ring_dict)
            visual_elements.append(ring)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for element in visual_elements:
            vis.add_geometry(element)

        opt = vis.get_render_option()
        opt.background_color = np.asarray(self.cmap.background)  # Charcoal grey background
        vis.run()
        vis.destroy_window()
