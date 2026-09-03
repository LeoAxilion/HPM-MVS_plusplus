#ifndef HPM_MVS_NEAREST_PLANE_PRIOR_H_
#define HPM_MVS_NEAREST_PLANE_PRIOR_H_

#include "main.h"

#include <cstddef>
#include <vector>

struct HNSWPlanarPriorStats {
    HNSWPlanarPriorStats()
        : input_support_count(0),
          unique_support_count(0),
          duplicate_support_count(0),
          mapped_duplicate_support_count(0),
          valid_support_count(0),
          invalid_support_count(0),
          reliable_pixel_count(0),
          queried_pixel_count(0),
          assigned_pixel_count(0),
          invalid_projected_depth_count(0) {}

    std::size_t input_support_count;
    std::size_t unique_support_count;
    std::size_t duplicate_support_count;
    std::size_t mapped_duplicate_support_count;
    std::size_t valid_support_count;
    std::size_t invalid_support_count;
    std::size_t reliable_pixel_count;
    std::size_t queried_pixel_count;
    std::size_t assigned_pixel_count;
    std::size_t invalid_projected_depth_count;
};

struct HNSWPlanarPriorResult {
    // Sparse camera-frame planes. A positive value in plane_labels is the
    // corresponding zero-based index in this vector plus one.
    std::vector<float4> plane_parameters;
    cv::Mat_<float> plane_labels;
    cv::Mat_<float> prior_depths;
    std::vector<cv::Point> mapped_support_points;
    HNSWPlanarPriorStats stats;
};

// Builds a full-resolution prior label map from reliable points sampled on an
// arbitrary resolution grid. full_normals are expected to be in world
// coordinates, matching normals.dmb produced by HPM. Returned planes are in
// the reference-camera coordinate frame and can be passed directly to
// HPM::CudaPlanarPriorInitialization together with plane_labels.
HNSWPlanarPriorResult BuildHNSWPlanarPrior(
    const std::vector<cv::Point>& support_points,
    const cv::Size& support_grid_size,
    const cv::Mat_<float>& full_depths,
    const cv::Mat_<cv::Vec3f>& full_normals,
    const cv::Mat_<cv::Vec3b>& full_bgr,
    const Camera& camera,
    float depth_min,
    float depth_max);

#endif  // HPM_MVS_NEAREST_PLANE_PRIOR_H_
