#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <sstream>
#include <unordered_map>
#include <random>
#include<chrono>

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

    return error.norm();  // Return the Euclidean distance (L2 norm) of the error
}

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
    for (int i = 0; i < num_observations + 1; ++i) {
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

// Gaussian error function
double gaussian_error_function(double x, double mu = 0.0, double sigma = 1.0) {
    return std::exp(-std::pow(x - mu, 2) / (2 * std::pow(sigma, 2)));
}

Eigen::Matrix3d compute_sample_covariance(const std::vector<Eigen::Vector3d>& samples) {
    if (samples.size() < 2) {
        throw std::runtime_error("Not enough samples to compute covariance.");
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& sample : samples) {
        mean += sample;
    }
    mean /= samples.size();

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& sample : samples) {
        Eigen::Vector3d diff = sample - mean;
        covariance += diff * diff.transpose();
    }
    covariance /= (samples.size() - 1);

    return covariance;
}


void refine_variance(PointStruct& pointStruct) {
    auto start = std::chrono::high_resolution_clock::now();  // Start timing

    const int MAX_ITERS = 10000;
    int counter = 0;
    std::vector<Eigen::Vector3d> samples;
    samples.push_back(pointStruct.mean); // Initial sample
    double previous_fx = 1.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < MAX_ITERS; ++i) {
        Eigen::Vector3d next_point = samples.back() + Eigen::Vector3d(dist(gen), dist(gen), dist(gen)).cwiseProduct(pointStruct.initial_variance.diagonal());

        std::vector<double> errors;
        for (size_t j = 0; j < pointStruct.cameras.size(); ++j) {
            double error = calculate_reprojection_error(pointStruct.cameras[j], next_point, pointStruct.observations[j]);
            errors.push_back(error);
        }

        double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

        if (mean_error >= 1.0) continue;

        double accept = std::uniform_real_distribution<>(0.0, 1.0)(gen);
        double current_fx = gaussian_error_function(mean_error);
        double ratio = std::min(1.0, current_fx / previous_fx);

        if (accept < ratio) {
            samples.push_back(next_point);
            previous_fx = current_fx;
            counter++;
        }

        if (counter > 250) {
            // Update variance
            // TODO: Implement variance update
            pointStruct.initial_variance = compute_sample_covariance(samples);
            counter = 0;
        }
    }
  
    pointStruct.initial_variance = compute_sample_covariance(samples);
    auto end = std::chrono::high_resolution_clock::now();    // End timing
    std::chrono::duration<double> elapsed = end - start;     // Calculate elapsed time
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
    // std::cout << "Final Covariance Matrix:\n" << pointStruct.initial_variance << "\n";
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



void SaveToBALFile(const std::string& filename, const std::vector<Camera>& cameras, const std::vector<PointStruct>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Calculate the total number of observations
    int num_observations = 0;
    for (const auto& point : points) {
        num_observations += point.cameras.size();
    }

    // Write the header
    file << cameras.size() << " " << points.size() << " " << num_observations << "\n";

    // Write the observations
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points[i].cameras.size(); ++j) {
            file << j << " " << i << " " << points[i].observations[j].transpose() << "\n";
        }
    }

    // Write camera parameters
    for (const auto& camera : cameras) {
        file << camera.rotation[0] << "\n" << camera.rotation[1] << "\n" << camera.rotation[2] << "\n";
        file << camera.translation[0] << "\n" << camera.translation[1] << "\n" << camera.translation[2] << "\n";
        file << camera.focal_length << "\n" << camera.cx << "\n" << camera.cy << "\n";
    }

    // Write 3D world points
    for (const auto& point : points) {
        file << point.mean[0] << "\n" << point.mean[1] << "\n" << point.mean[2] << "\n";
    }

    file.close();
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

void savePointStructToFile(const PointStruct& pointStruct, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Save the mean
    file << "Mean:\n" << pointStruct.mean.transpose() << "\n";

    // Save the variance
    file << "Variance:\n" << pointStruct.initial_variance << "\n";

    file.close();
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

    SaveToBALFile("text.txt", cameras, points);

    assess_reprojection_errors(mcmc);
    
    int idx = 0;
    refine_variance(mcmc.points[idx]);
    std::cout << "Final Covariance Matrix:\n" << mcmc.points[idx].initial_variance << "\n";
    savePointStructToFile(mcmc.points[idx], "output.txt");

    // mcmc.Run();

    return 0; // Successful execution
}
