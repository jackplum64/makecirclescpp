#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <memory>
#include <fstream>
#include <unistd.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>



class Circle
{
public:
    Circle(float radius, int x = -1, int y = -1) : radius(radius), x(x), y(y) {}

    void setPoint(int x, int y) {
        this->x = x;
        this->y = y;
    }

    void print() const {
        std::cout << "Radius: " << radius << ", Center: ";
        if (x == -1 && y == -1) {
            std::cout << "Not Set";
        } else {
            std::cout << "(" << x << ", " << y << ")";
        }
        std::cout << std::endl;
    }

    int getX() const { return x; }
    int getY() const { return y; }
    float getRadius() const { return radius; }

private:
    float radius;
    int x, y;
};



class Grid
{
public:
    Grid(int width, int height, int minCellSize)
        : width(width), height(height), minCellSize(minCellSize) {
        // Calculate the number of cells in each dimension
        numCellsX = (width + minCellSize - 1) / minCellSize;
        numCellsY = (height + minCellSize - 1) / minCellSize;
        cellSizeX = static_cast<float>(width) / numCellsX;
        cellSizeY = static_cast<float>(height) / numCellsY;

        bins.resize(numCellsX * numCellsY);
    }

    void addCircle(const Circle& circle) {
        int cellIndex = getCellIndex(circle.getX(), circle.getY());
        if (cellIndex != -1) {
            bins[cellIndex].push_back(circle);
        }
    }

    std::vector<Circle> getBin(int x, int y, int distance = 0) const {
        std::vector<Circle> result;
        int cellIndex = getCellIndex(x, y);

        if (cellIndex == -1) {
            return result;
        }

        std::vector<std::pair<int, int>> offsets = { {0, 0} };
        if (distance == 1) {
            offsets = { {0, 0}, {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1} };
        }
        if (distance == 2) {
            offsets = {
                {0, 0}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
                {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
                {-2, 0}, {-1, 0}, {1, 0}, {2, 0},
                {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1},
                {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2}
            };
        }

        for (const auto& offset : offsets) {
            int newX = x + offset.first * static_cast<int>(cellSizeX);
            int newY = y + offset.second * static_cast<int>(cellSizeY);
            int newCellIndex = getCellIndex(newX, newY);
            if (newCellIndex != -1) {
                result.insert(result.end(), bins[newCellIndex].begin(), bins[newCellIndex].end());
            }
        }

        return result;
    }

    void print() const {
        for (int i = 0; i < numCellsY; i++) {
            for (int j = 0; j < numCellsX; j++) {
                std::cout << "Grid [" << i << ", " << j << "] Contains:\n";
                const std::vector<Circle>& circles = bins[i * numCellsX + j];
                for (const auto& circle : circles) {
                    circle.print();
                }
            }
        }
    }

    int getNumCellsX() const { return numCellsX; }
    int getNumCellsY() const { return numCellsY; }
    float getCellSizeX() const { return cellSizeX; }
    float getCellSizeY() const { return cellSizeY; }

private:
    int width, height, minCellSize;
    int numCellsX, numCellsY;
    float cellSizeX, cellSizeY;
    std::vector<std::vector<Circle>> bins;

    int getCellIndex(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return -1;
        }
        int cellX = static_cast<int>(x / cellSizeX);
        int cellY = static_cast<int>(y / cellSizeY);
        return cellY * numCellsX + cellX;
    }
};



class CircleGroup
{
public:
    CircleGroup(int width, int height, float mean, float mean_delta, float std_dev, float std_dev_delta, int count, std::vector<Circle> excludeCircles = {})
        : width(width), height(height), mean(mean), mean_delta(mean_delta), std_dev(std_dev), std_dev_delta(std_dev_delta), count(count), grid(nullptr), excludeCircles(excludeCircles) {
        bool goodPack = false;
        while (!goodPack) {
            // Clear circles vector and reset grid
            circles.clear();
            grid.reset();

            generateCircles();
            goodPack = assignCircleCenters();
        }
    }

    const std::vector<Circle>& getCircles() const {
        return circles;
    }

    float getMaxRadius() const {
        return maxRadius;
    }

    std::pair<float, float> getStats() const {
        return { calculatedMean, calculatedStdDev };
    }

    void print() {
        for (const auto& circle : getCircles()) {
            circle.print();
        }
        std::cout << "Maximum Radius: " << getMaxRadius() << std::endl;
        std::cout << "Mean Radius: " << calculatedMean << std::endl;
        std::cout << "Standard Deviation: " << calculatedStdDev << std::endl;
    }

    const Grid& getGrid() const {
        return *grid;
    }

private:
    int width, height;
    float mean, mean_delta;
    float std_dev, std_dev_delta;
    float maxRadius;
    int count;
    float calculatedMean, calculatedStdDev;
    std::vector<Circle> circles;
    std::unique_ptr<Grid> grid;
    std::vector<Circle> excludeCircles;

    void generateCircles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(mean, std_dev);

        bool withinDelta = false;

        while (!withinDelta) {
            circles.clear();
            maxRadius = 0.0f;
            for (int i = 0; i < count; ++i) {
                float radius = dist(gen);
                if (radius < 0) radius = -radius; // Ensure radius is non-negative
                circles.emplace_back(radius);
                if (radius > maxRadius) {
                    maxRadius = radius;
                }
            }

            // Calculate mean and standard deviation of the generated radii
            float sum = 0;
            for (const auto& circle : circles) {
                sum += circle.getRadius();
            }
            calculatedMean = sum / circles.size();

            float varianceSum = 0;
            for (const auto& circle : circles) {
                varianceSum += std::pow(circle.getRadius() - calculatedMean, 2);
            }
            calculatedStdDev = std::sqrt(varianceSum / circles.size());

            // Check if the calculated mean and standard deviation are within the specified deltas
            withinDelta = std::abs(calculatedMean - mean) <= mean_delta &&
                          std::abs(calculatedStdDev - std_dev) <= std_dev_delta;
        }

        std::sort(circles.begin(), circles.end(), [](const Circle& a, const Circle& b) {
        return a.getRadius() > b.getRadius();
        });
    }

    bool assignCircleCenters() {
        grid = std::make_unique<Grid>(width, height, static_cast<int>(std::ceil(maxRadius)));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distX(0, width);
        std::uniform_int_distribution<> distY(0, height);

        int numCircles = 0;
        int maxcount = 50000;

        for (auto& circle : circles) {
            bool placed = false;
            int counter = 0;

            if (numCircles / count >= 0.5) {
                double exp = (count * 3.1415926535 * calculatedMean) / (height * width);
                maxcount = (numCircles / count) * pow(50000, exp);
            }

            while (!placed) {
                counter++;
                if (counter > maxcount) {
                    //std::cout << "RESET" << std::endl;
                    return false;
                }
                int x = distX(gen);
                int y = distY(gen);

                if (x - circle.getRadius() >= 0 && x + circle.getRadius() <= width &&
                    y - circle.getRadius() >= 0 && y + circle.getRadius() <= height) {
                } else {
                    continue;
                }
                if (doesOverlap(x, y, circle.getRadius())) {
                    continue;
                }
                if (doesOverlapFully(x, y, circle.getRadius(), excludeCircles)) {
                    continue;
                }
                circle.setPoint(x, y);
                grid->addCircle(circle);
                placed = true;
                numCircles++;
            }
        }
        return true;
    }

    bool doesOverlap(int newX, int newY, float newRadius) const {
        std::vector<Circle> nearbyCircles = grid->getBin(newX, newY, 2);
        for (const auto& circle : nearbyCircles) {
            float minDistance = circle.getRadius() + newRadius;

            if (std::abs(circle.getX() - newX) > minDistance) {
                continue;
            }
            if (std::abs(circle.getY() - newY) > minDistance) {
                continue;
            }

            if (std::pow(circle.getX() - newX, 2) + std::pow(circle.getY() - newY, 2) <= std::pow(minDistance, 2)) {
                return true;
            }
        }
        return false;
    }

    bool doesOverlapFully(int newX, int newY, float newRadius, const std::vector<Circle>& excludeCircles) const {
        if (excludeCircles.empty()) {
            return false;
        }
        for (const auto& circle : excludeCircles) {
            float distanceSquared = std::pow(circle.getX() - newX, 2) + std::pow(circle.getY() - newY, 2);
            float radiusDifference = std::abs(circle.getRadius() - newRadius);
            float radiusSum = circle.getRadius() + newRadius;

            // Check if the new circle is fully contained within the existing circle
            if (distanceSquared <= std::pow(circle.getRadius() - newRadius, 2)) {
                return true;
            }
            // Check if the existing circle is fully contained within the new circle
            if (distanceSquared <= std::pow(newRadius - circle.getRadius(), 2)) {
                return true;
            }
        }
        return false;
    }
};




void saveImage(const cv::Mat& image, const std::string& outputDir, int index) {
    // Create output directory if it does not exist
    cv::utils::fs::createDirectory(outputDir);

    // Save the original image with index
    std::string originalFilename = outputDir + "/circles_" + std::to_string(index) + ".png";
    cv::imwrite(originalFilename, image);

    // Create a blurred version of the image
    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(9, 9), 0);

    // Save the blurred image with index
    std::string blurredFilename = outputDir + "/circles_blurred_" + std::to_string(index) + ".png";
    cv::imwrite(blurredFilename, blurredImage);
}


void drawGridLines(cv::Mat& image, const Grid& grid) {
    int numCellsX = grid.getNumCellsX();
    int numCellsY = grid.getNumCellsY();
    float cellSizeX = grid.getCellSizeX();
    float cellSizeY = grid.getCellSizeY();

    for (int i = 0; i <= numCellsX; ++i) {
        cv::line(image, cv::Point(i * cellSizeX, 0), cv::Point(i * cellSizeX, 1000), cv::Scalar(0, 0, 255));
    }
    for (int i = 0; i <= numCellsY; ++i) {
        cv::line(image, cv::Point(0, i * cellSizeY), cv::Point(1000, i * cellSizeY), cv::Scalar(0, 0, 255));
    }
}


void drawCircles(cv::Mat& image, const std::vector<Circle>& circles, cv::Scalar color) {
    for (const auto& circle : circles) {
        cv::circle(image, cv::Point(circle.getX(), circle.getY()), circle.getRadius(), color, -1);
        cv::circle(image, cv::Point(circle.getX(), circle.getY()), 1, color, -1);
    }
}


std::string getExecutablePath() {
    char buffer[1024];
    ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (count != -1) {
        buffer[count] = '\0';
        return std::string(buffer);
    }
    return "";
}


void loadConfig(const std::string& filename, int& width, int& height, int& numOutputs, float& g1Mean, float& g1Mean_delta, float& g1Std_dev, float& g1Std_dev_delta, int& g1Count, cv::Scalar& g1Color, float& g2Mean, float& g2Mean_delta, float& g2Std_dev, float& g2Std_dev_delta, int& g2Count, cv::Scalar& g2Color) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace from the beginning and end of the line
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        // Skip empty lines and lines starting with '#'
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                if (key == "width") width = std::stoi(value);
                else if (key == "height") height = std::stoi(value);
                else if (key == "numOutputs") numOutputs = std::stoi(value);
                else if (key == "group1_mean") g1Mean = std::stof(value);
                else if (key == "group1_mean_delta") g1Mean_delta = std::stof(value);
                else if (key == "group1_std_dev") g1Std_dev = std::stof(value);
                else if (key == "group1_std_dev_delta") g1Std_dev_delta = std::stof(value);
                else if (key == "group1_count") g1Count = std::stoi(value);
                else if (key == "group1_color_r") g1Color[2] = std::stoi(value);
                else if (key == "group1_color_g") g1Color[1] = std::stoi(value);
                else if (key == "group1_color_b") g1Color[0] = std::stoi(value);
                else if (key == "group2_mean") g2Mean = std::stof(value);
                else if (key == "group2_mean_delta") g2Mean_delta = std::stof(value);
                else if (key == "group2_std_dev") g2Std_dev = std::stof(value);
                else if (key == "group2_std_dev_delta") g2Std_dev_delta = std::stof(value);
                else if (key == "group2_count") g2Count = std::stoi(value);
                else if (key == "group2_color_r") g2Color[2] = std::stoi(value);
                else if (key == "group2_color_g") g2Color[1] = std::stoi(value);
                else if (key == "group2_color_b") g2Color[0] = std::stoi(value);
            }
        }
    }
}


std::string formatFloat(float value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}


std::queue<int> taskQueue;
std::mutex queueMutex;
std::condition_variable condVar;
bool stopThreads = false;

void workerFunction(int width, int height, float g1Mean, float g1Mean_delta, float g1Std_dev, float g1Std_dev_delta, int g1Count, cv::Scalar g1Color,
                    float g2Mean, float g2Mean_delta, float g2Std_dev, float g2Std_dev_delta, int g2Count, cv::Scalar g2Color, const std::string& outputDir, int totalTasks) {
    while (true) {
        int taskIndex;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condVar.wait(lock, []{ return !taskQueue.empty() || stopThreads; });

            if (stopThreads && taskQueue.empty()) {
                return;
            }

            taskIndex = taskQueue.front();
            taskQueue.pop();
        }

        // Generate the circles and images
        CircleGroup g1(width, height, g1Mean, g1Mean_delta, g1Std_dev, g1Std_dev_delta, g1Count);
        const std::vector<Circle>& g1Circles = g1.getCircles();
        CircleGroup g2(width, height, g2Mean, g2Mean_delta, g2Std_dev, g2Std_dev_delta, g2Count, g1Circles);
        const std::vector<Circle>& g2Circles = g2.getCircles();

        cv::Mat g1Image = cv::Mat::zeros(height, width, CV_8UC3);
        cv::Mat g2Image = cv::Mat::zeros(height, width, CV_8UC3);
        drawCircles(g1Image, g1Circles, g1Color);
        drawCircles(g2Image, g2Circles, g2Color);

        cv::Mat finalImage;
        cv::addWeighted(g1Image, 1, g2Image, 1, 0, finalImage, -1);
        saveImage(finalImage, outputDir, taskIndex);

        std::cout << "Finished " << taskIndex + 1 << "/" << totalTasks << std::endl;

        if (taskIndex + 1 == totalTasks) {
            stopThreads = true;
            condVar.notify_all();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3 || std::string(argv[1]) != "-config") {
        std::cerr << "Usage: " << argv[0] << " -config <config_path>" << std::endl;
        return 1;
    }
    std::string config_path = argv[2];

    auto start = std::chrono::high_resolution_clock::now();

    int width, height, numOutputs;
    float g1Mean, g1Mean_delta, g1Std_dev, g1Std_dev_delta;
    int g1Count;
    cv::Scalar g1Color;
    float g2Mean, g2Mean_delta, g2Std_dev, g2Std_dev_delta;
    int g2Count;
    cv::Scalar g2Color;

    loadConfig(config_path, width, height, numOutputs, g1Mean, g1Mean_delta, g1Std_dev, g1Std_dev_delta, g1Count, g1Color, g2Mean, g2Mean_delta, g2Std_dev, g2Std_dev_delta, g2Count, g2Color);

    std::cout << "Generating " << numOutputs << " images with parameters:\n"
              << "GROUP 1 - " << "Mean: " << g1Mean << ", Mean delta: " << g1Mean_delta << ", Std dev: " << g1Std_dev << ", Std dev delta: " << g1Std_dev_delta << ", Count: " << g1Count << "\n"
              << "GROUP 2 - " << "Mean: " << g2Mean << ", Mean delta: " << g2Mean_delta << ", Std dev: " << g2Std_dev << ", Std dev delta: " << g2Std_dev_delta << ", Count: " << g2Count << std::endl;
    std::cout << std::endl;

    std::string executablePath = getExecutablePath();
    std::string parentDir = cv::utils::fs::getParent(executablePath);
    std::string baseOutputDir = parentDir + "/output";

    // Create the base output directory if it doesn't exist
    cv::utils::fs::createDirectory(baseOutputDir);

    // Create a directory name based on the statistics with two decimal places
    std::string outputDir = baseOutputDir + "/output_"
                            + "g1_mean_" + formatFloat(g1Mean) + "_std_" + formatFloat(g1Std_dev)
                            + "_g1_count_" + std::to_string(g1Count) // Add group 1 count
                            + "_g2_mean_" + formatFloat(g2Mean) + "_std_" + formatFloat(g2Std_dev)
                            + "_g2_count_" + std::to_string(g2Count); // Add group 2 count

    // Create the detailed output directory
    cv::utils::fs::createDirectory(outputDir);

    // Populate the task queue
    for (int i = 0; i < numOutputs; ++i) {
        taskQueue.push(i);
    }

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    for (int i = 0; i < numThreads; ++i) {
        workers.emplace_back(workerFunction, width, height, g1Mean, g1Mean_delta, g1Std_dev, g1Std_dev_delta, g1Count, g1Color, g2Mean, g2Mean_delta, g2Std_dev, g2Std_dev_delta, g2Count, g2Color, outputDir, numOutputs);
    }

    for (auto& worker : workers) {
        worker.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::endl;
    std::cout << "Saved " << numOutputs << " images to " << outputDir << std::endl;
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}