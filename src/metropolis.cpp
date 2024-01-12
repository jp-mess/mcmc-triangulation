#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <sstream>
#include <unordered_map>

// Define a struct for camera parameters
struct Camera {
    Eigen::Vector3d rotation;  // Angle-axis rotation
    Eigen::Vector3d translation;
    double focal_length;
    double cx, cy;
};

// Define a struct for each point in the point cloud
struct PointStruct {
    Eigen::Vector3d mean;
    Eigen::Matrix3d initial_variance;
    std::vector<Camera> cameras;  // Cameras that viewed this point
    std::vector<Eigen::Vector2d> observations;
};

// Class for Metropolis-Hastings MCMC
class MCMCSimulation {
public:
    std::vector<PointStruct> points;

    void AddPoint(const PointStruct& point) {
        points.push_back(point);
    }

    // Method to perform MCMC sampling
    void Run() {
        // Implement MCMC algorithm here
    }
};

void LoadPoints(const std::string& filename, std::vector<Camera>& cameras, std::vector<PointStruct>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int num_cameras, num_points, num_observations;
    file >> num_cameras >> num_points >> num_observations;

    // Resize camera and point vectors
    cameras.resize(num_cameras);
    points.resize(num_points);

    // Skip observation data
    for (int i = 0; i < num_observations; ++i) {
        std::string line;
        std::getline(file, line); // Read and discard the line
    }

    // Read camera parameters
    for (int i = 0; i < num_cameras; ++i) {
        for (int j = 0; j < 9; ++j) { // 9 parameters per camera
            if (j < 3) file >> cameras[i].rotation[j];    // Rotation
            else if (j < 6) file >> cameras[i].translation[j - 3]; // Translation
            else if (j == 6) file >> cameras[i].focal_length; // Focal length
            else if (j == 7) file >> cameras[i].cx; // cx
            else if (j == 8) file >> cameras[i].cy; // cy
        }
    }

    // Read world points
    for (int i = 0; i < num_points; ++i) {
        file >> points[i].mean[0] >> points[i].mean[1] >> points[i].mean[2];
        points[i].initial_variance = Eigen::Matrix3d::Identity() * 0.01;
    }
}

void LoadMCMC(const std::string& filename, const std::vector<Camera>& cameras, std::vector<PointStruct>& points, MCMCSimulation& mcmc) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int num_cameras, num_points, num_observations;
    file >> num_cameras >> num_points >> num_observations;

    // Resize the points vector to accommodate all points
    points.resize(num_points);

    // Read observation data and link cameras and observations to points
    for (int i = 0; i < num_observations; ++i) {
        int camera_idx, point_idx;
        double image_x, image_y;
        file >> camera_idx >> point_idx >> image_x >> image_y;

        if (camera_idx < num_cameras && point_idx < num_points) {
            // Add the camera to the point's observed cameras
            points[point_idx].cameras.push_back(cameras[camera_idx]);

            // Add the observed pixel coordinates
            points[point_idx].observations.emplace_back(image_x, image_y);
        }
    }

    // Set the points in MCMC
    mcmc.points = points;
}

double calculate_reprojection_error(const Camera& camera, const Eigen::Vector3d& point, const Eigen::Vector2d& observed_pixel) {
    // Convert angle-axis rotation to rotation matrix
    Eigen::AngleAxisd rotation_vector(camera.rotation.norm(), camera.rotation.normalized());
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();

    // Transform the point from world coordinates to camera coordinates
    Eigen::Vector3d point_camera = rotation_matrix.transpose() * (point - camera.translation);

    // Project the point onto the image plane
    // Assuming the camera points along the z-axis and z>0 is in front of the camera
    double x_projected = camera.focal_length * (point_camera.x() / point_camera.z()) + camera.cx;
    double y_projected = camera.focal_length * (point_camera.y() / point_camera.z()) + camera.cy;

    // Calculate the reprojection error
    Eigen::Vector2d projected_pixel(x_projected, y_projected);
    Eigen::Vector2d error = observed_pixel - projected_pixel;

    std::cout << projected_pixel << std::endl;
    std::cout << observed_pixel << std::endl;
    std::cout << std::endl;

    return error.norm();  // Return the Euclidean distance (L2 norm) of the error
}

// Function to assess reprojection errors for all points in MCMC
void assess_reprojection_errors(MCMCSimulation& mcmc) {
    for (const auto& point_struct : mcmc.points) {
        const auto& point = point_struct.mean;
        for (size_t i = 0; i < point_struct.cameras.size(); ++i) {
            const auto& camera = point_struct.cameras[i];
            const auto& observed_pixel = point_struct.observations[i];

            double error = calculate_reprojection_error(camera, point, observed_pixel);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_bal_file.txt>" << std::endl;
        return 1; // Exit with error code
    }

    std::string filename = argv[1]; // The first argument is the path to the BAL file

    MCMCSimulation mcmc;
    std::vector<Camera> cameras;
    std::vector<PointStruct> points;
    LoadPoints(filename, cameras, points);
    LoadMCMC(filename, cameras, points, mcmc);

    assess_reprojection_errors(mcmc);
    
    mcmc.Run();

    return 0; // Successful execution
}
