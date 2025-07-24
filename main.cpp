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



class Sphere {
public:
    Sphere(float radius_px, int x= -1, int y= -1, int z= -1)
      : r(radius_px), x(x), y(y), z(z) {}

    void setCenter(int xi, int yi, int zi) {
        x = xi;  y = yi;  z = zi;
    }

    float radius() const { return r; }
    int   getX()    const { return x; }
    int   getY()    const { return y; }
    int   getZ()    const { return z; }

    void print() const {
        std::cout
          << "r=" << r << "px at ("
          << x << "," << y << "," << z << ")\n";
    }

private:
    float r;   // radius (px)
    int   x,y,z;
};



class Grid3D {
public:
    Grid3D(int width_px, int height_px, int depth_px, int minCellSize_px)
      : W(width_px), H(height_px), D(depth_px)
    {
        // # of cells in each dim
        nx = (W + minCellSize_px - 1) / minCellSize_px;
        ny = (H + minCellSize_px - 1) / minCellSize_px;
        nz = (D + minCellSize_px - 1) / minCellSize_px;

        // actual cell sizes (float px)
        cx = float(W)/nx;
        cy = float(H)/ny;
        cz = float(D)/nz;

        bins.resize(nx*ny*nz);
    }

    // add one sphere into its containing bin
    void add(const Sphere& s) {
        int idx = cellIndex(s.getX(), s.getY(), s.getZ());
        if (idx>=0) bins[idx].push_back(s);
    }

    // gather all spheres within Manhattan-bin‐distance `dist` of the bin containing (x,y,z)
    std::vector<Sphere> getNeighbors(int x, int y, int z, int dist=1) const {
        std::vector<Sphere> out;
        int cx0 = int(x/cx), cy0 = int(y/cy), cz0 = int(z/cz);

        for(int dz=-dist; dz<=dist; ++dz){
            int cz_ = cz0+dz;
            if(cz_<0||cz_>=nz) continue;
            for(int dy=-dist; dy<=dist; ++dy){
                int cy_ = cy0+dy;
                if(cy_<0||cy_>=ny) continue;
                for(int dx=-dist; dx<=dist; ++dx){
                    int cx_ = cx0+dx;
                    if(cx_<0||cx_>=nx) continue;
                    int idx = cz_*ny*nx + cy_*nx + cx_;
                    // append
                    out.insert(out.end(),
                               bins[idx].begin(),
                               bins[idx].end());
                }
            }
        }
        return out;
    }

private:
    int W,H,D;       // volume size (px)
    int nx,ny,nz;    // bin counts
    float cx,cy,cz;  // bin size (px)
    std::vector<std::vector<Sphere>> bins;

    // map world coords → flat bin index
    int cellIndex(int x, int y, int z) const {
        if(x<0||x>=W||y<0||y>=H||z<0||z>=D) return -1;
        int ix = int(x/cx), iy=int(y/cy), iz=int(z/cz);
        return iz*ny*nx + iy*nx + ix;
    }
};



class SphereGroup {
public:
    SphereGroup(int W, int H, int D,
                float mean_r, float mean_tol,
                float std_r,  float std_tol,
                int count,
                const std::vector<Sphere>& exclude = {})
      : W(W), H(H), D(D),
        mu(mean_r), muTol(mean_tol),
        sigma(std_r), sigmaTol(std_tol),
        N(count), excl(exclude)
    {
        // keep drawing until stats are within tol, then place
        do {
            sampleRadii();
            computeStats();
        } while(!statsOK());

        // sort largest→smallest
        std::sort(sph.begin(), sph.end(),
                  [](auto &a, auto &b){
                    return a.radius()>b.radius();
                  });

        placeAll();
    }

    const std::vector<Sphere>& spheres() const { return sph; }

private:
    int W,H,D, N;
    float mu, muTol, sigma, sigmaTol;
    std::vector<Sphere> sph, excl;
    float meanGen, stdGen, maxR;

    // 3D grid for overlap checks
    std::unique_ptr<Grid3D> grid;

    // sample N radii from Normal(μ,σ)
    void sampleRadii() {
        std::random_device rd;
        std::mt19937       gen(rd());
        std::normal_distribution<> dist(mu, sigma);

        sph.clear();
        maxR = 0;
        for(int i=0; i<N; ++i){
            float r = dist(gen);
            if(r<0) r = -r;
            sph.emplace_back(r);
            if(r>maxR) maxR = r;
        }
    }

    // compute meanGen, stdGen
    void computeStats(){
        float sum=0;
        for(auto &s: sph) sum += s.radius();
        meanGen = sum/N;

        float var=0;
        for(auto &s: sph){
            float d = s.radius()-meanGen;
            var += d*d;
        }
        stdGen = std::sqrt(var/N);
    }

    bool statsOK() const {
        return std::fabs(meanGen-mu)<=muTol &&
               std::fabs(stdGen-sigma)<=sigmaTol;
    }

    // place spheres one‐by‐one using rejection sampling
    void placeAll(){
        grid = std::make_unique<Grid3D>(W,H,D, int(std::ceil(maxR)));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dx(0,W),
                                  dy(0,H),
                                  dz(0,D);

        for(auto &s : sph){
            bool placed = false;
            int attempts = 0, maxAtt=500000;
            while(!placed && attempts<maxAtt){
                ++attempts;
                int x = dx(gen), y=dy(gen), z=dz(gen);
                float r = s.radius();
                // boundary
                if(x<r||x>W-r||y<r||y>H-r||z<r||z>D-r) continue;

                if(overlaps(x,y,z,r)) continue;
                if(overlapExcl(x,y,z,r)) continue;

                s.setCenter(x,y,z);
                grid->add(s);
                placed = true;
            }
            if(!placed){
                std::cerr<<"Failed to place all spheres\n";
                std::exit(1);
            }
        }
    }

    // any existing sphere within r1+r2?
    bool overlaps(int x,int y,int z,float rnew) const {
        auto neigh = grid->getNeighbors(x,y,z,2);
        for(auto &o: neigh){
            float R = o.radius()+rnew;
            float dx = o.getX()-x,
                  dy = o.getY()-y,
                  dz = o.getZ()-z;
            if(dx*dx+dy*dy+dz*dz <= R*R)
                return true;
        }
        return false;
    }

    // check against `excl` list (fully contained)
    bool overlapExcl(int x,int y,int z,float rnew) const {
        for(auto &o: excl){
            float dx = o.getX()-x,
                  dy = o.getY()-y,
                  dz = o.getZ()-z;
            float d2 = dx*dx+dy*dy+dz*dz;
            float dr = std::fabs(o.radius()-rnew);
            if(d2 <= dr*dr) return true;
        }
        return false;
    }
};

void writeXYZR(const std::string& filepath, const std::vector<Sphere>& spheres) {
    std::ofstream ofs(filepath);
    if (!ofs) {
        std::cerr << "Error: could not open '" << filepath << "' for writing\n";
        return;
    }
    for (const auto& s : spheres) {
        ofs
          << s.getX()    << ' '  // X coordinate (px)
          << s.getY()    << ' '  // Y coordinate (px)
          << s.getZ()    << ' '  // Z coordinate (px)
          << s.radius()  << '\n';// radius (px)
    }
}

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


std::string getExecutablePath() {
    char buffer[1024];
    ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (count != -1) {
        buffer[count] = '\0';
        return std::string(buffer);
    }
    return "";
}


void loadConfig(const std::string &filename,
                int &W, int &H, int &D,
                int &numOutputs,
                float &g1Mean, float &g1MeanDelta,
                float &g1Std, float &g1StdDelta,
                int &g1Count, cv::Scalar &g1Color,
                float &g2Mean, float &g2MeanDelta,
                float &g2Std, float &g2StdDelta,
                int &g2Count, cv::Scalar &g2Color,
                bool &enableGroup2)
{
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open config: " + filename);
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0]=='#') continue;
        std::istringstream iss(line);
        std::string key, val;
        if (std::getline(iss, key, '=') && std::getline(iss, val)) {
            if (key=="width")        W = std::stoi(val);
            else if (key=="height")  H = std::stoi(val);
            else if (key=="depth")   D = std::stoi(val);
            else if (key=="numOutputs") numOutputs = std::stoi(val);
            else if (key=="group1_mean")           g1Mean      = std::stof(val);
            else if (key=="group1_mean_delta")     g1MeanDelta = std::stof(val);
            else if (key=="group1_std_dev")        g1Std       = std::stof(val);
            else if (key=="group1_std_dev_delta")  g1StdDelta  = std::stof(val);
            else if (key=="group1_count")          g1Count     = std::stoi(val);
            else if (key=="group1_color_r")        g1Color[2]  = std::stoi(val);
            else if (key=="group1_color_g")        g1Color[1]  = std::stoi(val);
            else if (key=="group1_color_b")        g1Color[0]  = std::stoi(val);
            else if (key=="group2_mean")           g2Mean      = std::stof(val);
            else if (key=="group2_mean_delta")     g2MeanDelta = std::stof(val);
            else if (key=="group2_std_dev")        g2Std       = std::stof(val);
            else if (key=="group2_std_dev_delta")  g2StdDelta  = std::stof(val);
            else if (key=="group2_count")          g2Count     = std::stoi(val);
            else if (key=="group2_color_r")        g2Color[2]  = std::stoi(val);
            else if (key=="group2_color_g")        g2Color[1]  = std::stoi(val);
            else if (key=="group2_color_b")        g2Color[0]  = std::stoi(val);
            else if (key=="enable_group2")         enableGroup2 = (val=="1"||val=="true");
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

void workerFunction(int W, int H, int D,
                    float g1Mean, float g1MeanDelta,
                    float g1Std,  float g1StdDelta,
                    int g1Count,  cv::Scalar g1Color,
                    float g2Mean, float g2MeanDelta,
                    float g2Std,  float g2StdDelta,
                    int g2Count,  cv::Scalar g2Color,
                    bool enableGroup2,
                    const std::string &outDir,
                    int totalTasks)
{
    while (true) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(queueMutex);
            condVar.wait(lk, []{ return !taskQueue.empty() || stopThreads; });
            if (stopThreads && taskQueue.empty()) return;
            idx = taskQueue.front(); taskQueue.pop();
        }

        SphereGroup sg1(W,H,D, g1Mean,g1MeanDelta, g1Std,g1StdDelta, g1Count);
        auto s1 = sg1.spheres();

        std::vector<Sphere> all = s1;
        if (enableGroup2) {
            SphereGroup sg2(W,H,D, g2Mean,g2MeanDelta, g2Std,g2StdDelta, g2Count, s1);
            auto s2 = sg2.spheres();
            all.insert(all.end(), s2.begin(), s2.end());
        }

        // Draw 2D projection (z ignored) onto image
        cv::Mat img = cv::Mat::zeros(H, W, CV_8UC3);
        for (auto &s : all) {
            cv::circle(img, cv::Point(s.getX(), s.getY()), int(s.radius()),
                       (&s==&all[0] ? g1Color : g2Color), -1);
        }

        cv::utils::fs::createDirectory(outDir);
        std::string imgPath = outDir + "/img_" + std::to_string(idx) + ".png";
        cv::imwrite(imgPath, img);

        std::string xyzrPath = outDir + "/spheres_" + std::to_string(idx) + ".xyzr";
        writeXYZR(xyzrPath, all);

        std::cout << "Done " << (idx+1) << "/" << totalTasks << std::endl;
        if (idx+1 == totalTasks) {
            stopThreads = true;
            condVar.notify_all();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3 || std::string(argv[1]) != "-config") {
        std::cerr << "Usage: " << argv[0] << " -config <path>\n";
        return 1;
    }
    std::string cfg = argv[2];

    int W=0, H=0, D=0, numOut=0;
    float g1Mean=0, g1MD=0, g1Std=0, g1SD=0;
    int g1Count=0; cv::Scalar g1Color;
    float g2Mean=0, g2MD=0, g2Std=0, g2SD=0;
    int g2Count=0; cv::Scalar g2Color;
    bool enableGroup2 = true;

    loadConfig(cfg,
               W, H, D,
               numOut,
               g1Mean,g1MD, g1Std,g1SD, g1Count,g1Color,
               g2Mean,g2MD, g2Std,g2SD, g2Count,g2Color,
               enableGroup2);

    for (int i = 0; i < numOut; ++i)
        taskQueue.push(i);

    std::string exe = cv::utils::fs::getParent(argv[0]);
    std::string outDir = exe + "/output";

    int threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) {
        workers.emplace_back(workerFunction,
                             W,H,D,
                             g1Mean,g1MD, g1Std,g1SD, g1Count,g1Color,
                             g2Mean,g2MD, g2Std,g2SD, g2Count,g2Color,
                             enableGroup2,
                             outDir, numOut);
    }
    condVar.notify_all();
    for (auto &t : workers) t.join();

    return 0;
}