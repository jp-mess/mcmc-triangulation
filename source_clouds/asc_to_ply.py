import open3d as o3d

def convert_asc_to_ply(asc_file_path):
    # Load the point cloud from the .asc file
    point_cloud = o3d.io.read_point_cloud(asc_file_path, format='xyz')

    # Replace the file extension from .asc to .ply
    ply_file_path = asc_file_path.replace('.asc', '.ply')

    # Save the point cloud in .ply format
    o3d.io.write_point_cloud(ply_file_path, point_cloud)

    return ply_file_path

# Example usage
asc_file_path = 'guanyin.asc'
ply_file_path = convert_asc_to_ply(asc_file_path)
print(f"Converted file saved as: {ply_file_path}")
