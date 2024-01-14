
def create_wireframe_box(center, extents, rotation_matrix):
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
    wireframe.paint_uniform_color([0.184, 0.310, 0.310])  # Electro-blue color
    return wireframe

def create_frustum(camera_extrinsic, dim=0.2):
    camera_position = camera_extrinsic[:3, 3]
    rotation_matrix = camera_extrinsic[:3, :3]
    extents = np.array([dim, dim, dim * 1.25])
    wireframe_box = create_wireframe_box(camera_position, extents, rotation_matrix)
    return wireframe_box

def create_direction_line(camera_extrinsic, length=5):
    camera_position = camera_extrinsic[:3, 3]
    camera_direction = camera_extrinsic[:3, 2]
    end_point = camera_position + camera_direction * length
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([camera_position, end_point])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.paint_uniform_color([0.0, 1.0, 0.0])  # Green color
    return line

def add_camera_visualizations(vis, cameras):
    for camera in cameras:
        frustum = create_frustum(camera['extrinsic'])
        line = create_direction_line(camera['extrinsic'])
        vis.add_geometry(frustum)
        vis.add_geometry(line)

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

def generate_line_segments(data_list):
    import numpy as np

    line_segments = []

    for data in data_list:
        mean = data['mean']
        variance = data['variance']

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(variance)

        # Find the eigenvector corresponding to the largest eigenvalue
        max_eigval_index = np.argmax(eigenvalues)
        max_eigvec = eigenvectors[:, max_eigval_index]
        std_dev = np.sqrt(eigenvalues[max_eigval_index])

        # Calculate the start and end points of the line segment
        start_point = mean - max_eigvec * std_dev / 2
        end_point = mean + max_eigvec * std_dev / 2

        line_segments.append([start_point, end_point])

    return line_segments

def read_means_and_covariances(file_path):
    import numpy as np
    from tqdm import tqdm

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_list = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'Mean:':
            mean = np.array([float(x) for x in lines[i + 1].strip().split()])
            i += 2  # Skip to the next line after the mean

        if i < len(lines) and lines[i].strip() == 'Variance:':
            variance_lines = lines[i + 1:i + 4]
            variance = np.array([[float(x) for x in line.split()] for line in variance_lines])
            i += 4  # Skip to the next line after the variance

            data_list.append({'mean': mean, 'variance': variance})
        else:
            i += 1

    return data_list



def read_point_struct_from_file(file_path):
    import numpy as np
    data = {"mean": None, "variance": None}
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find and extract mean
    mean_index = lines.index("Mean:\n")
    mean_values = lines[mean_index + 1].strip().split()
    data["mean"] = np.array(mean_values, dtype=float)

    # Find and extract variance
    variance_index = lines.index("Variance:\n")
    variance_values = [line.strip().split() for line in lines[variance_index + 1:variance_index + 4]]
    data["variance"] = np.array(variance_values, dtype=float)

    return data



if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import open3d as o3d
  import numpy as np
  import geometry_utils

  #eigen_file = "small_covariances.txt"
  #bal_file = "small_angles.txt"

  eigen_file = "large_covariances.txt"
  bal_file = "large_angles.txt"

  cameras, points = geometry_utils.load_bal_problem_file(bal_file)


  data_list = read_means_and_covariances(eigen_file)
  line_segments = generate_line_segments(data_list)

  # Example usage
  vis = o3d.visualization.Visualizer()
  vis.create_window()

  # Assuming 'line_segments' is the list of line segments from the eigenvector plotting
  # Add these line segments to the visualization
  for segment in line_segments:
      lines = o3d.geometry.LineSet()
      lines.points = o3d.utility.Vector3dVector(segment)
      lines.lines = o3d.utility.Vector2iVector([[0, 1]])
      lines.paint_uniform_color([0.184, 0.310, 0.310])  # Electro-blue color
      vis.add_geometry(lines)

  # Assuming 'cameras' is a list of dictionaries with camera extrinsics
  add_camera_visualizations(vis, cameras)

  vis.run()
  vis.destroy_window()



