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
            int attempts = 0, maxAtt=10000000;
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
    bool overlapExcl(int x, int y, int z, float rnew) const {
        for (const auto &o : excl) {
            // distances in pixels
            float dx   = o.getX() - x;      // px
            float dy   = o.getY() - y;      // px
            float dz   = o.getZ() - z;      // px
            // sum of radii in pixels
            float Rsum = o.radius() + rnew; // px
            // if centers closer than sum of radii → overlap
            if (dx*dx + dy*dy + dz*dz <= Rsum*Rsum)
                return true;
        }
        return false;
    }
};


struct SphereMode {
    float  mean_r;       // px
    float  mean_tol;     // px
    float  std_r;        // px
    float  std_tol;      // px
    int    count;
    cv::Scalar color;    // BGR
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
                int &modeCount,
                std::vector<SphereMode> &modes)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open config: " + filename);

    // first pass: load simple scalars, including mode_count
    std::map<std::string,std::string> kv;
    for (std::string line; std::getline(file, line); ) {
        auto trim = [](std::string &s){
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n")+1);
        };
        trim(line);
        if (line.empty() || line[0]=='#') continue;
        auto eq = line.find('=');
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq+1);
        trim(key); trim(val);
        kv[key] = val;
    }

    W          = std::stoi(kv["width"]);
    H          = std::stoi(kv["height"]);
    D          = std::stoi(kv["depth"]);
    numOutputs = std::stoi(kv["numOutputs"]);
    modeCount  = std::stoi(kv["mode_count"]);

    modes.clear();
    modes.reserve(modeCount);
    for (int i = 1; i <= modeCount; ++i) {
        SphereMode m{};
        m.mean_r   = std::stof(kv["mode" + std::to_string(i) + "_mean"]);
        m.mean_tol = std::stof(kv["mode" + std::to_string(i) + "_mean_delta"]);
        m.std_r    = std::stof(kv["mode" + std::to_string(i) + "_std_dev"]);
        m.std_tol  = std::stof(kv["mode" + std::to_string(i) + "_std_dev_delta"]);
        m.count    = std::stoi(kv["mode" + std::to_string(i) + "_count"]);
        // load colors as R,G,B
        int r = std::stoi(kv["mode" + std::to_string(i) + "_color_r"]);
        int g = std::stoi(kv["mode" + std::to_string(i) + "_color_g"]);
        int b = std::stoi(kv["mode" + std::to_string(i) + "_color_b"]);
        m.color = cv::Scalar(b, g, r);
        modes.push_back(m);
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
                    std::shared_ptr<std::vector<SphereMode>> modesPtr,
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

        // We’ll keep a growing list of “already placed” to exclude
        std::vector<Sphere> placedExcl;
        // And collect per-mode spheres for drawing:
        std::vector<std::vector<Sphere>> allGroups;
        allGroups.reserve(modesPtr->size());

        // sample each mode in turn, excluding previously placed
        for (auto &m : *modesPtr) {
            SphereGroup sg(W, H, D,
                           m.mean_r, m.mean_tol,
                           m.std_r,  m.std_tol,
                           m.count,
                           placedExcl);
            auto groupSpheres = sg.spheres();
            allGroups.push_back(groupSpheres);
            // add this mode’s spheres to the exclusion list
            placedExcl.insert(placedExcl.end(),
                              groupSpheres.begin(),
                              groupSpheres.end());
        }

        // draw 2D projection
        cv::Mat img = cv::Mat::zeros(H, W, CV_8UC3);
        for (size_t gi = 0; gi < allGroups.size(); ++gi) {
            const auto &grp = allGroups[gi];
            const auto &col = (*modesPtr)[gi].color;
            for (auto &s : grp) {
                cv::circle(img,
                           cv::Point(s.getX(), s.getY()),
                           int(std::ceil(s.radius())),
                           col,
                           -1);
            }
        }

        std::vector<Sphere> all;
        for (auto &grp : allGroups) {
            all.insert(all.end(), grp.begin(), grp.end());
        }

        cv::utils::fs::createDirectory(outDir);
        std::string imgPath  = outDir + "/img_"     + std::to_string(idx) + ".png";
        std::string xyzrPath = outDir + "/spheres_" + std::to_string(idx) + ".xyzr";

        cv::imwrite(imgPath, img);
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

    int W = 0, H = 0, D = 0, numOut = 0, modeCount = 0;
    std::vector<SphereMode> modes;

    loadConfig(cfg, W, H, D, numOut, modeCount, modes);
    for (int i = 0; i < numOut; ++i)
        taskQueue.push(i);

    auto modesPtr = std::make_shared<std::vector<SphereMode>>(modes);
    std::string exe    = cv::utils::fs::getParent(argv[0]);
    std::string outDir = exe + "/output";

    int threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back(workerFunction,
                             W, H, D,
                             modesPtr,
                             outDir, numOut);
    }
    condVar.notify_all();
    for (auto &t : workers) t.join();

    return 0;
}