#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "iomanip"

// #include<direct.h>  //windows mkdir
#include <sys/stat.h> //linux mkdir
#include <sys/types.h> //linux mkdir


#define MAX_IMAGES 256
#define JBU_NUM 2

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    // Longest image side after the configured initial downscale.
    int initial_max_side = 0;
    int num_downscale = 0;
    // Longest image side for the current pyramid level.
    int cur_max_side = 0;
    std::shared_ptr<class ScaledImageCache> scaled_image_cache;
};

// Per-ProcessProblem cache for prepared, layer-scaled source images.
class ScaledImageCache {
public:
    explicit ScaledImageCache(size_t max_images);
    void Retain(const std::vector<int>& image_ids);
    void SetOriginalSize(int image_id, const cv::Size& size);
    cv::Size OriginalSize(int image_id) const;
    cv::Mat Get(int image_id, const std::string& scaled_image_path);
    void Clear();
    void ResetStats();
    size_t CacheHits() const { return cache_hits; }
    size_t CacheMisses() const { return cache_misses; }
    double DiskReadSeconds() const { return disk_read_seconds; }

private:
    size_t capacity;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    double disk_read_seconds = 0.0;
    std::unordered_map<int, cv::Mat> images;
    std::unordered_map<int, cv::Size> original_sizes;
};

std::string ScaledImagePath(const std::string& dense_folder, int image_id,
    int target_max_side);
int PyramidMaxSideForImage(const std::vector<Problem>& problems, int image_id);

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

#define M_PI 3.14159265358979323846

#endif // _MAIN_H_
