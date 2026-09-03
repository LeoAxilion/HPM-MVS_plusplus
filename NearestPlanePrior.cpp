#include "NearestPlanePrior.h"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

const std::size_t kNearestSupportCount = 5;
const std::size_t kHnswM = 16;
const std::size_t kHnswEfConstruction = 200;
const std::size_t kHnswEfSearch = 64;
const std::size_t kHnswSeed = 100;
const float kNormalEpsilon = 1.0e-6f;
const float kProjectionEpsilon = 1.0e-8f;

struct SupportRecord {
    float position[2];
    cv::Point mapped_point;
    cv::Vec3f lab;
    float4 plane;
};

bool IsFinite(const float value)
{
    return std::isfinite(value) != 0;
}

bool IsFinite(const cv::Vec3f& value)
{
    return IsFinite(value[0]) && IsFinite(value[1]) && IsFinite(value[2]);
}

bool IsFinite(const float4& value)
{
    return IsFinite(value.x) && IsFinite(value.y) &&
           IsFinite(value.z) && IsFinite(value.w);
}

void ValidateFullResolutionInputs(
    const cv::Mat_<float>& full_depths,
    const cv::Mat_<cv::Vec3f>& full_normals,
    const cv::Mat_<cv::Vec3b>& full_bgr,
    const Camera& camera,
    const float depth_min,
    const float depth_max)
{
    if (full_depths.empty()) {
        throw std::invalid_argument("full_depths must not be empty");
    }
    if (full_normals.size() != full_depths.size() ||
        full_bgr.size() != full_depths.size()) {
        throw std::invalid_argument(
            "full_depths, full_normals, and full_bgr must have the same size");
    }
    if (!IsFinite(depth_min) || !IsFinite(depth_max) ||
        depth_min >= depth_max) {
        throw std::invalid_argument("invalid planar-prior depth range");
    }
    if (!IsFinite(camera.K[0]) || !IsFinite(camera.K[2]) ||
        !IsFinite(camera.K[4]) || !IsFinite(camera.K[5]) ||
        std::fabs(camera.K[0]) <= kProjectionEpsilon ||
        std::fabs(camera.K[4]) <= kProjectionEpsilon) {
        throw std::invalid_argument("invalid reference-camera intrinsics");
    }
}

cv::Point MapSupportPointToFullResolution(
    const cv::Point& point,
    const cv::Size& support_grid_size,
    const cv::Size& full_size,
    float* mapped_x,
    float* mapped_y)
{
    const float scale_x = static_cast<float>(full_size.width) /
                          static_cast<float>(support_grid_size.width);
    const float scale_y = static_cast<float>(full_size.height) /
                          static_cast<float>(support_grid_size.height);

    // Match OpenCV's resize pixel-center convention.
    *mapped_x = (static_cast<float>(point.x) + 0.5f) * scale_x - 0.5f;
    *mapped_y = (static_cast<float>(point.y) + 0.5f) * scale_y - 0.5f;

    const int x = std::max(0, std::min(full_size.width - 1,
                                      cvRound(*mapped_x)));
    const int y = std::max(0, std::min(full_size.height - 1,
                                      cvRound(*mapped_y)));
    return cv::Point(x, y);
}

bool MakeCameraPlane(
    const cv::Point& support,
    const float depth,
    const cv::Vec3f& world_normal,
    const Camera& camera,
    float4* plane)
{
    if (!IsFinite(depth) || !IsFinite(world_normal)) {
        return false;
    }

    // Camera.R maps world-frame vectors into the reference-camera frame.
    float nx = camera.R[0] * world_normal[0] +
               camera.R[1] * world_normal[1] +
               camera.R[2] * world_normal[2];
    float ny = camera.R[3] * world_normal[0] +
               camera.R[4] * world_normal[1] +
               camera.R[5] * world_normal[2];
    float nz = camera.R[6] * world_normal[0] +
               camera.R[7] * world_normal[1] +
               camera.R[8] * world_normal[2];

    const float normal_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (!IsFinite(normal_norm) || normal_norm <= kNormalEpsilon) {
        return false;
    }
    nx /= normal_norm;
    ny /= normal_norm;
    nz /= normal_norm;

    const float x = depth *
        (static_cast<float>(support.x) - camera.K[2]) / camera.K[0];
    const float y = depth *
        (static_cast<float>(support.y) - camera.K[5]) / camera.K[4];
    const float distance = -(nx * x + ny * y + nz * depth);

    *plane = make_float4(nx, ny, nz, distance);
    return IsFinite(*plane);
}

float ProjectDepth(
    const float4& plane,
    const int x,
    const int y,
    const Camera& camera)
{
    const float ray_x =
        (static_cast<float>(x) - camera.K[2]) / camera.K[0];
    const float ray_y =
        (static_cast<float>(y) - camera.K[5]) / camera.K[4];
    const float denominator =
        plane.x * ray_x + plane.y * ray_y + plane.z;
    if (!IsFinite(denominator) ||
        std::fabs(denominator) <= kProjectionEpsilon) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return -plane.w / denominator;
}

float SquaredLabDistance(const cv::Vec3f& first, const cv::Vec3f& second)
{
    const float d0 = first[0] - second[0];
    const float d1 = first[1] - second[1];
    const float d2 = first[2] - second[2];
    return d0 * d0 + d1 * d1 + d2 * d2;
}

}  // namespace

HNSWPlanarPriorResult BuildHNSWPlanarPrior(
    const std::vector<cv::Point>& support_points,
    const cv::Size& support_grid_size,
    const cv::Mat_<float>& full_depths,
    const cv::Mat_<cv::Vec3f>& full_normals,
    const cv::Mat_<cv::Vec3b>& full_bgr,
    const Camera& camera,
    const float depth_min,
    const float depth_max)
{
    ValidateFullResolutionInputs(
        full_depths, full_normals, full_bgr, camera, depth_min, depth_max);

    HNSWPlanarPriorResult result;
    result.stats.input_support_count = support_points.size();
    result.plane_labels =
        cv::Mat_<float>::zeros(full_depths.rows, full_depths.cols);
    result.prior_depths =
        cv::Mat_<float>::zeros(full_depths.rows, full_depths.cols);

    if (support_points.empty()) {
        return result;
    }
    if (support_grid_size.width <= 0 || support_grid_size.height <= 0) {
        throw std::invalid_argument(
            "support_grid_size must be positive when support points exist");
    }

    cv::Mat_<cv::Vec3f> bgr_float;
    full_bgr.convertTo(bgr_float, CV_32FC3, 1.0 / 255.0);
    cv::Mat_<cv::Vec3f> lab_image;
    cv::cvtColor(bgr_float, lab_image, cv::COLOR_BGR2Lab);
    bgr_float.release();

    std::set<std::pair<int, int> > seen_supports;
    std::vector<cv::Point> unique_supports;
    unique_supports.reserve(support_points.size());
    for (std::size_t i = 0; i < support_points.size(); ++i) {
        const cv::Point& point = support_points[i];
        if (seen_supports.insert(std::make_pair(point.x, point.y)).second) {
            unique_supports.push_back(point);
        }
    }
    result.stats.unique_support_count = unique_supports.size();
    result.stats.duplicate_support_count =
        result.stats.input_support_count - result.stats.unique_support_count;

    std::set<std::pair<int, int> > seen_mapped_supports;
    std::vector<SupportRecord> records;
    records.reserve(unique_supports.size());

    for (std::size_t i = 0; i < unique_supports.size(); ++i) {
        const cv::Point& point = unique_supports[i];
        if (point.x < 0 || point.x >= support_grid_size.width ||
            point.y < 0 || point.y >= support_grid_size.height) {
            ++result.stats.invalid_support_count;
            continue;
        }

        SupportRecord record;
        record.mapped_point = MapSupportPointToFullResolution(
            point, support_grid_size, full_depths.size(),
            &record.position[0], &record.position[1]);

        const std::pair<int, int> mapped_key = std::make_pair(
            record.mapped_point.x, record.mapped_point.y);
        if (!seen_mapped_supports.insert(mapped_key).second) {
            ++result.stats.mapped_duplicate_support_count;
            ++result.stats.invalid_support_count;
            continue;
        }

        const float depth = full_depths(
            record.mapped_point.y, record.mapped_point.x);
        if (depth < depth_min || depth > depth_max ||
            !MakeCameraPlane(
                record.mapped_point, depth,
                full_normals(record.mapped_point.y, record.mapped_point.x),
                camera, &record.plane)) {
            ++result.stats.invalid_support_count;
            continue;
        }

        const float projected_support_depth = ProjectDepth(
            record.plane, record.mapped_point.x, record.mapped_point.y,
            camera);
        if (!IsFinite(projected_support_depth) ||
            projected_support_depth < depth_min ||
            projected_support_depth > depth_max) {
            ++result.stats.invalid_support_count;
            continue;
        }

        record.lab = lab_image(record.mapped_point.y, record.mapped_point.x);
        if (!IsFinite(record.lab)) {
            ++result.stats.invalid_support_count;
            continue;
        }
        records.push_back(record);
    }

    result.stats.valid_support_count = records.size();
    if (records.empty()) {
        return result;
    }

    result.plane_parameters.reserve(records.size());
    result.mapped_support_points.reserve(records.size());
    std::vector<unsigned char> reliable_pixels(
        static_cast<std::size_t>(full_depths.rows) *
            static_cast<std::size_t>(full_depths.cols),
        static_cast<unsigned char>(0));

    hnswlib::L2Space space(2);
    hnswlib::HierarchicalNSW<float> index(
        &space, records.size(), kHnswM, kHnswEfConstruction, kHnswSeed);
    index.setEf(std::max(kNearestSupportCount, kHnswEfSearch));

    for (std::size_t i = 0; i < records.size(); ++i) {
        index.addPoint(
            records[i].position, static_cast<hnswlib::labeltype>(i));
        result.plane_parameters.push_back(records[i].plane);
        result.mapped_support_points.push_back(records[i].mapped_point);
        const std::size_t center =
            static_cast<std::size_t>(records[i].mapped_point.y) *
                static_cast<std::size_t>(full_depths.cols) +
            static_cast<std::size_t>(records[i].mapped_point.x);
        reliable_pixels[center] = static_cast<unsigned char>(1);
    }

    result.stats.reliable_pixel_count = records.size();
    const std::size_t pixel_count =
        static_cast<std::size_t>(full_depths.rows) *
        static_cast<std::size_t>(full_depths.cols);
    result.stats.queried_pixel_count =
        pixel_count - result.stats.reliable_pixel_count;

    const std::size_t query_k =
        std::min(kNearestSupportCount, records.size());
    std::size_t assigned_count = 0;
    std::size_t invalid_projection_count = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    reduction(+:assigned_count, invalid_projection_count)
#endif
    for (int y = 0; y < full_depths.rows; ++y) {
        float* label_row = result.plane_labels.ptr<float>(y);
        float* prior_depth_row = result.prior_depths.ptr<float>(y);
        const cv::Vec3f* lab_row = lab_image.ptr<cv::Vec3f>(y);

        for (int x = 0; x < full_depths.cols; ++x) {
            const std::size_t center =
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(full_depths.cols) +
                static_cast<std::size_t>(x);
            if (reliable_pixels[center] != 0) {
                continue;
            }

            const float query[2] = {
                static_cast<float>(x), static_cast<float>(y)};
            std::priority_queue<
                std::pair<float, hnswlib::labeltype> > nearest =
                index.searchKnn(query, query_k);

            hnswlib::labeltype best_label =
                std::numeric_limits<hnswlib::labeltype>::max();
            float best_color_distance = std::numeric_limits<float>::max();
            float best_spatial_distance = std::numeric_limits<float>::max();

            while (!nearest.empty()) {
                const float spatial_distance = nearest.top().first;
                const hnswlib::labeltype label = nearest.top().second;
                nearest.pop();
                if (label >= records.size()) {
                    continue;
                }

                const float color_distance = SquaredLabDistance(
                    lab_row[x], records[label].lab);
                if (color_distance < best_color_distance ||
                    (color_distance == best_color_distance &&
                     (spatial_distance < best_spatial_distance ||
                      (spatial_distance == best_spatial_distance &&
                       label < best_label)))) {
                    best_label = label;
                    best_color_distance = color_distance;
                    best_spatial_distance = spatial_distance;
                }
            }

            if (best_label ==
                std::numeric_limits<hnswlib::labeltype>::max()) {
                ++invalid_projection_count;
                continue;
            }

            const float prior_depth = ProjectDepth(
                records[best_label].plane, x, y, camera);
            if (!IsFinite(prior_depth) || prior_depth < depth_min ||
                prior_depth > depth_max) {
                ++invalid_projection_count;
                continue;
            }

            label_row[x] = static_cast<float>(best_label + 1);
            prior_depth_row[x] = prior_depth;
            ++assigned_count;
        }
    }

    result.stats.assigned_pixel_count = assigned_count;
    result.stats.invalid_projected_depth_count =
        invalid_projection_count;
    return result;
}
