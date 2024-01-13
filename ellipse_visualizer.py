
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
  data = read_point_struct_from_file("src/build/output.txt")
  ax = plot_cov_ellipse(data["variance"], data["mean"])
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  plt.show()




