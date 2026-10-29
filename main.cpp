#include "main.h"
#include "HPM.h"
#include <cstdio>
#include <dirent.h>
#include <stdexcept>
#include <chrono>
#include <cerrno>
#include <unordered_set>

class ExecutionTimer {
public:
	ExecutionTimer() : start_(std::chrono::steady_clock::now()) {}

	~ExecutionTimer() {
		const std::chrono::duration<double> elapsed =
			std::chrono::steady_clock::now() - start_;
		std::cout << "Total execution time: " << std::fixed << std::setprecision(2)
			<< elapsed.count() << " seconds" << std::endl;
	}

private:
	std::chrono::steady_clock::time_point start_;
};

ScaledImageCache::ScaledImageCache(size_t max_images) : capacity(max_images) {
	if (capacity == 0) throw std::invalid_argument("scaled-image cache capacity must be positive");
}

void ScaledImageCache::Retain(const std::vector<int>& image_ids) {
	std::unordered_set<int> keep(image_ids.begin(), image_ids.end());
	if (keep.size() > capacity) {
		throw std::runtime_error("ProcessProblem image count exceeds scaled-image cache capacity");
	}
	for (auto it = images.begin(); it != images.end();) {
		if (keep.find(it->first) == keep.end()) it = images.erase(it);
		else ++it;
	}
}

void ScaledImageCache::SetOriginalSize(int image_id, const cv::Size& size) {
	original_sizes[image_id] = size;
}

cv::Size ScaledImageCache::OriginalSize(int image_id) const {
	auto it = original_sizes.find(image_id);
	if (it == original_sizes.end()) {
		throw std::runtime_error("Original dimensions are missing for scaled image " +
			std::to_string(image_id));
	}
	return it->second;
}

cv::Mat ScaledImageCache::Get(int image_id, const std::string& scaled_image_path) {
	auto it = images.find(image_id);
	if (it != images.end()) {
		++cache_hits;
		return it->second;
	}
	if (images.size() >= capacity) {
		throw std::runtime_error("Scaled-image cache is full; Retain() must run before loading a view");
	}
	const auto started = std::chrono::steady_clock::now();
	cv::Mat image = cv::imread(scaled_image_path, cv::IMREAD_COLOR);
	if (image.empty()) {
		throw std::runtime_error("Cannot read prepared scaled image: " + scaled_image_path);
	}
	disk_read_seconds += std::chrono::duration<double>(
		std::chrono::steady_clock::now() - started).count();
	++cache_misses;
	images.emplace(image_id, image);
	return image;
}

void ScaledImageCache::Clear() {
	images.clear();
}

void ScaledImageCache::ResetStats() {
	cache_hits = 0;
	cache_misses = 0;
	disk_read_seconds = 0.0;
}

std::string ScaledImagePath(const std::string& dense_folder, int image_id,
	int target_max_side) {
	std::stringstream image_path;
	image_path << dense_folder << "/HPM_MVS_plusplus/2333_" << std::setw(8)
		<< std::setfill('0') << image_id << "/scaled_image_maxside"
		<< target_max_side << ".png";
	return image_path.str();
}

int PyramidMaxSideForImage(const std::vector<Problem>& problems, int image_id) {
	for (const Problem& problem : problems) {
		if (problem.ref_image_id == image_id) return problem.cur_max_side;
	}
	throw std::runtime_error("No pyramid scale found for source image " +
		std::to_string(image_id));
}

static void EnsureDirectory(const std::string& path) {
	if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
		throw std::runtime_error("Cannot create directory: " + path);
	}
}

std::vector<std::string> PrepareScaleImages(const std::string& dense_folder,
	const std::vector<Problem>& problems, ScaledImageCache& cache) {
	std::unordered_set<int> image_ids;
	for (const Problem& problem : problems) {
		image_ids.insert(problem.ref_image_id);
		image_ids.insert(problem.src_image_ids.begin(), problem.src_image_ids.end());
	}
	std::vector<int> ordered_ids(image_ids.begin(), image_ids.end());
	std::sort(ordered_ids.begin(), ordered_ids.end());
	std::vector<std::string> files;
	files.reserve(ordered_ids.size());
	const std::string image_folder = dense_folder + "/images";
	const auto started = std::chrono::steady_clock::now();
	for (size_t i = 0; i < ordered_ids.size(); ++i) {
		const int image_id = ordered_ids[i];
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0')
			<< image_id << ".jpg";
		cv::Mat image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
		if (image.empty()) throw std::runtime_error("Cannot read image: " + image_path.str());
		const cv::Size original_size = image.size();
		const int target_max_side = PyramidMaxSideForImage(problems, image_id);
		cache.SetOriginalSize(image_id, original_size);
		if (std::max(image.cols, image.rows) > target_max_side) {
			const float factor = std::min(
				static_cast<float>(target_max_side) / image.cols,
				static_cast<float>(target_max_side) / image.rows);
			const int new_cols = std::max(1, static_cast<int>(std::round(image.cols * factor)));
			const int new_rows = std::max(1, static_cast<int>(std::round(image.rows * factor)));
			cv::Mat scaled;
			cv::resize(image, scaled, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
			image = scaled;
		}
		std::stringstream result_folder_stream;
		result_folder_stream << dense_folder << "/HPM_MVS_plusplus/2333_"
			<< std::setw(8) << std::setfill('0') << image_id;
		const std::string result_folder = result_folder_stream.str();
		EnsureDirectory(result_folder);
		const std::string scaled_path = ScaledImagePath(
			dense_folder, image_id, target_max_side);
		if (!cv::imwrite(scaled_path, image,
			std::vector<int>{cv::IMWRITE_PNG_COMPRESSION, 1})) {
			throw std::runtime_error("Cannot write prepared scaled image: " + scaled_path);
		}
		files.push_back(scaled_path);
		if ((i + 1) % 20 == 0 || i + 1 == ordered_ids.size()) {
			std::cout << "Prepared scaled images " << i + 1 << "/" << ordered_ids.size()
				<< " for current pyramid layer." << std::endl;
		}
	}
	std::cout << "Prepared " << files.size() << " scaled images in "
		<< std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
		<< " s." << std::endl;
	return files;
}

void RemoveScaleImageFiles(const std::vector<std::string>& files) {
	size_t removed = 0;
	for (const std::string& file : files) {
		if (std::remove(file.c_str()) == 0) ++removed;
	}
	std::cout << "Removed " << removed << " temporary scaled images." << std::endl;
}

void GenerateSampleList(const std::string& dense_folder, std::vector<Problem>& problems)
{
	std::string cluster_list_path = dense_folder + std::string("/pair.txt");

	problems.clear();

	std::ifstream file(cluster_list_path);

	int num_images;
	file >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.src_image_ids.clear();
		file >> problem.ref_image_id;

		int num_src_images;
		file >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			file >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

int ComputeMultiScaleSettings(const std::string& dense_folder, std::vector<Problem>& problems, const int initial_image_scale)
{
	int max_num_downscale = -1;
	int size_bound = 1000;
	std::string image_folder = dense_folder + std::string("/images");

	size_t num_images = problems.size();

	for (size_t i = 0; i < num_images; ++i) {
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);

		const int original_max_side = std::max(image_uint.rows, image_uint.cols);
		const int initial_max_side = std::max(
			1, (original_max_side + initial_image_scale / 2) / initial_image_scale);
		problems[i].initial_max_side = initial_max_side;

		int k = 0;
		int pyramid_max_side = initial_max_side;
		while (pyramid_max_side > size_bound) {
			pyramid_max_side /= 2;
			k++;
		}

		if (k > max_num_downscale) {
			max_num_downscale = k;
		}

		problems[i].num_downscale = k;
	}

	return max_num_downscale;
}

void ProcessProblem(const std::string& dense_folder, const std::vector<Problem>& problems, const int idx, bool geom_consistency, bool prior_consistency, bool hierarchy, bool mand_consistency, int image_scale, bool multi_geometrty = false, int hpm_scale_distance = 0, bool show_medium_result = false)
{
	const Problem problem = problems[idx];
	std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	if (!problem.scaled_image_cache) {
		throw std::runtime_error("ProcessProblem is missing its scaled-image cache");
	}
	std::vector<int> active_image_ids;
	active_image_ids.reserve(problem.src_image_ids.size() + 1);
	active_image_ids.push_back(problem.ref_image_id);
	active_image_ids.insert(active_image_ids.end(), problem.src_image_ids.begin(),
		problem.src_image_ids.end());
	problem.scaled_image_cache->Retain(active_image_ids);
	cudaSetDevice(0);
	std::stringstream result_path;
	result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
	std::string result_folder = result_path.str();
	// mkdir(result_folder.c_str());	//windows
	mkdir(result_folder.c_str(), 0755);	//linux

	HPM hpm;
	if (geom_consistency) {
		hpm.SetGeomConsistencyParams(multi_geometrty);
	}
	if (hierarchy) {
		hpm.SetHierarchyParams();
	}

	hpm.InuputInitialization(dense_folder, problems, idx);
	hpm.CudaSpaceInitialization(dense_folder, problem);

	const int width = hpm.GetReferenceImageWidth();
	const int height = hpm.GetReferenceImageHeight();

	cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
	cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
	cv::Mat_<float>texture = cv::Mat::zeros(height, width, CV_32FC1);

	hpm.SetMandConsistencyParams(mand_consistency);

	cv::Mat Image_scaled_color;
	cv::Mat Canny_edge;
	const std::string scaled_ref_path = ScaledImagePath(
		dense_folder, problem.ref_image_id, problem.cur_max_side);
	Image_scaled_color = problem.scaled_image_cache->Get(
		problem.ref_image_id, scaled_ref_path);
	if (Image_scaled_color.cols != width || Image_scaled_color.rows != height) {
		throw std::runtime_error("Prepared reference image size does not match HPM camera size");
	}
	cv::Canny(Image_scaled_color, Canny_edge, 50, 150);
	std::cout << "Get Canny egdes down!" << std::endl;
	hpm.CudaCannyInitialization(Canny_edge);
	Image_scaled_color.release();
	Canny_edge.release();

	if (mand_consistency || prior_consistency) {
		hpm.CudaConfidenceInitialization(dense_folder, problems, idx);
	}
	if (!prior_consistency && !mand_consistency) {
		hpm.TextureInformationInitialization();
	}

	if (!prior_consistency) {
		if (mand_consistency) {
			std::cout << "Run Mandatory Consistency ..." << std::endl;
		}
		else if (geom_consistency) {
			std::cout << "Run Geometric Consistency ..." << std::endl;
		}
		else {
			std::cout << "Run Photometric Consistency ..." << std::endl;
		}
		hpm.RunPatchMatch();
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				float4 plane_hypothesis = hpm.GetPlaneHypothesis(center);
				depths(row, col) = plane_hypothesis.w;
				normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
				costs(row, col) = hpm.GetCost(center);
				if (!mand_consistency) {
					texture(row, col) = hpm.GetTexture(center);
				}
			}
		}
	}
	else if (prior_consistency) {
		std::cout << "Run Prior Consistency ..." << std::endl;
		hpm.SetPlanarPriorParams();

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
		std::string result_folder = result_path.str();

		std::string suffix = "/depths.dmb";
		if (multi_geometrty) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		std::string conf_path = result_folder + "/confidence.dmb";
		std::string cost_path = result_folder + "/costs.dmb";

		cv::Mat_<float>confidences;

		readDepthDmb(depth_path, depths);
		readNormalDmb(normal_path, normals);
		readDepthDmb(conf_path, confidences);
		readDepthDmb(cost_path, costs);


		if (hpm_scale_distance == 0) {
			const cv::Rect imageRC(0, 0, width, height);
			std::vector<cv::Point> support2DPoints;

			std::string texture_path = result_folder + "/texture" + std::to_string(image_scale) + ".dmb";
			cv::Mat_<float>textures;
			readDepthDmb(texture_path, textures);

			hpm.GetSupportPoints_Classify_Check(support2DPoints, costs, confidences, textures, 1);
			const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints);

			cv::Mat refImage = hpm.GetReferenceImage().clone();
			std::vector<cv::Mat> mbgr(3);
			mbgr[0] = refImage.clone();
			mbgr[1] = refImage.clone();
			mbgr[2] = refImage.clone();
			cv::Mat srcImage;
			cv::merge(mbgr, srcImage);
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
				}
			}
			if (show_medium_result) {
				std::string triangulation_path = result_folder + "/triangulation0.png";
				cv::imwrite(triangulation_path, srcImage);
			}

			refImage.release();
			mbgr.clear();
			mbgr.shrink_to_fit();
			srcImage.release();

			cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
			std::vector<float4> planeParams_tri;
			planeParams_tri.clear();

			uint32_t idx = 0;
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
					float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
					float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

					float max_edge_length = std::max(L01, std::max(L02, L12));
					float step = 1.0 / max_edge_length;

					for (float p = 0; p < 1.0; p += step) {
						for (float q = 0; q < 1.0 - p; q += step) {
							int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
							int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
							mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
						}
					}
					float4 n4 = hpm.GetPriorPlaneParams(triangle, depths);
					planeParams_tri.push_back(n4);
					idx++;
				}
			}

			cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
			for (int i = 0; i < width; ++i) {
				for (int j = 0; j < height; ++j) {
					if (mask_tri(j, i) > 0) {
						float d = hpm.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
						if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
							priordepths(j, i) = d;
						}
						else {
							mask_tri(j, i) = 0;
						}
					}
				}
			}
			if (show_medium_result) {
				std::string prior_path = result_folder + "/depths_prior0.dmb";
				writeDepthDmb(prior_path, priordepths);
			}
			priordepths.release();

			hpm.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
			hpm.CudaHypothesesReload(depths, costs, normals);
			hpm.RunPatchMatch();
			textures.release();
			mask_tri.release();
			planeParams_tri.clear();
			planeParams_tri.shrink_to_fit();
			support2DPoints.clear();
			support2DPoints.shrink_to_fit();
			planeParams_tri.clear();
			planeParams_tri.shrink_to_fit();
		}
		else if (hpm_scale_distance == 1 || hpm_scale_distance == 2) {
			float hpm_factor = 1.0 / (hpm_scale_distance * 2);
			int hpm_width = std::round(width * hpm_factor);
			int hpm_height = std::round(height * hpm_factor);

			cv::Mat_<float>depths_downsample;
			cv::Mat_<float>costs_downsample;
			cv::Mat_<float>confidences_downsample;

			cv::resize(depths, depths_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			cv::resize(costs, costs_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			cv::resize(confidences, confidences_downsample, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);



			std::string texture_path = result_folder + "/texture" + std::to_string(image_scale + hpm_scale_distance) + ".dmb";
			cv::Mat_<float>textures;
			readDepthDmb(texture_path, textures);

			const cv::Rect imageRC(0, 0, hpm_width, hpm_height);
			std::vector<cv::Point> support2DPoints;
			support2DPoints.clear();
			hpm.GetSupportPoints_Classify_Check(support2DPoints, costs_downsample, confidences_downsample, textures, hpm_factor);

			const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints);

			cv::Mat refImage;
			cv::resize(hpm.GetReferenceImage().clone(), refImage, cv::Size(hpm_width, hpm_height), 0, 0, cv::INTER_LINEAR);
			std::vector<cv::Mat> mbgr(3);
			mbgr[0] = refImage.clone();
			mbgr[1] = refImage.clone();
			mbgr[2] = refImage.clone();
			cv::Mat srcImage;
			cv::merge(mbgr, srcImage);
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
					cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
				}
			}
			if (show_medium_result) {
				std::string triangulation_path = result_folder + "/triangulation" + std::to_string(hpm_scale_distance) + ".png";
				cv::imwrite(triangulation_path, srcImage);
			}

			std::vector<float4>planeParams_tri;
			cv::Mat_<float> mask_tri = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC1);
			planeParams_tri.clear();
			uint32_t idx = 0;
			for (const auto triangle : triangles) {
				if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
					float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
					float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
					float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

					float max_edge_length = std::max(L01, std::max(L02, L12));
					float step = 1.0 / max_edge_length;

					for (float p = 0; p < 1.0; p += step) {
						for (float q = 0; q < 1.0 - p; q += step) {
							int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
							int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
							mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
						}
					}
					//renew the camera's parameters 
					float4 n4 = hpm.GetPriorPlaneParams_factor(triangle, depths_downsample, hpm_factor);
					planeParams_tri.push_back(n4);
					idx++;
				}
			}

			cv::Mat_<float>priordepths = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC1);
			cv::Mat_<cv::Vec3f>priornormals = cv::Mat::zeros(hpm_height, hpm_width, CV_32FC3);

			for (int i = 0; i < hpm_width; ++i) {
				for (int j = 0; j < hpm_height; ++j) {
					if (mask_tri(j, i) > 0) {
						float d = hpm.GetDepthFromPlaneParam_factor(planeParams_tri[mask_tri(j, i) - 1], i, j, hpm_factor);
						if (d <= hpm.GetMaxDepth() * 1.2f && d >= hpm.GetMinDepth() * 0.6f) {
							priordepths(j, i) = d;
							float4 tmp_n4 = hpm.TransformNormal(planeParams_tri[mask_tri(j, i) - 1]);
							priornormals(j, i)[0] = tmp_n4.x;
							priornormals(j, i)[1] = tmp_n4.y;
							priornormals(j, i)[2] = tmp_n4.z;
						}
						else {
							mask_tri(j, i) = 0;
						}
					}
				}
			}

			if (show_medium_result) {
				std::string depths_prior_path = result_folder + "/depths_prior" + std::to_string(hpm_scale_distance) + ".dmb";
				writeDepthDmb(depths_prior_path, priordepths);
			}

			cv::Mat_<float> image_float = hpm.GetReferenceImage();
			cv::Mat_<float>priordepths_upsample = cv::Mat::zeros(height, width, CV_32FC1);
			cv::Mat_ < cv::Vec3f >priornormals_upsample = cv::Mat::zeros(height, width, CV_32FC3);
			std::cout << "Running JBU..." << std::endl;
			hpm.JointBilateralUpsampling_prior(image_float, priordepths, priordepths_upsample, priornormals, priornormals_upsample);

			cv::Mat_<float>mask_tri_new = cv::Mat::zeros(height, width, CV_32FC1);
			float4* prior_planeParams = new float4[height * width];
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if (priordepths_upsample(j, i) <= hpm.GetMaxDepth() && priordepths_upsample(j, i) >= hpm.GetMinDepth()) {
						mask_tri_new(j, i) = 1;
						int center = j * width + i;
						float4 tmp_reload;
						tmp_reload.x = priornormals_upsample(j, i)[0];
						tmp_reload.y = priornormals_upsample(j, i)[1];
						tmp_reload.z = priornormals_upsample(j, i)[2];
						tmp_reload.w = priordepths_upsample(j, i);
						tmp_reload = hpm.TransformNormal2RefCam(tmp_reload);
						float depth_now = tmp_reload.w;
						int2 p = make_int2(i, j);
						tmp_reload.w = hpm.GetDistance2Origin(p, depth_now, tmp_reload);
						prior_planeParams[center] = tmp_reload;
					}
					else {
						mask_tri_new(j, i) = 0;
					}
				}
			}

			if (show_medium_result) {
				const std::string depths_prior_path = result_folder + "/depths_prior" + std::to_string(hpm_scale_distance) + "_upsample.dmb";
				writeDepthDmb(depths_prior_path, priordepths_upsample);
			}
			hpm.ReloadPlanarPriorInitialization(mask_tri_new, prior_planeParams);
			hpm.CudaHypothesesReload(depths, costs, normals);
			hpm.RunPatchMatch();

			refImage.release();
			mbgr.clear();
			mbgr.shrink_to_fit();
			srcImage.release();
			textures.release();
			support2DPoints.clear();
			support2DPoints.shrink_to_fit();
			depths_downsample.release();
			costs_downsample.release();
			confidences_downsample.release();
			priordepths.release();
			priornormals.release();
			image_float.release();
			priordepths_upsample.release();
			priornormals_upsample.release();
			mask_tri_new.release();
			delete(prior_planeParams);
		}

		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				float4 plane_hypothesis = hpm.GetPlaneHypothesis(center);
				depths(row, col) = plane_hypothesis.w;
				normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
				costs(row, col) = hpm.GetCost(center);
			}
		}

		hpm.CudaPlanarPriorRelease();
	}

	std::string suffix = "/depths.dmb";
	if (geom_consistency) {
		suffix = "/depths_geom.dmb";
	}
	std::string depth_path = result_folder + suffix;
	std::string normal_path = result_folder + "/normals.dmb";
	std::string cost_path = result_folder + "/costs.dmb";

	writeDepthDmb(depth_path, depths);
	writeNormalDmb(normal_path, normals);
	writeDepthDmb(cost_path, costs);
	if (!mand_consistency && !prior_consistency) {
		std::string texture_path = result_folder + "/texture" + std::to_string(image_scale) + ".dmb";
		writeDepthDmb(texture_path, texture);
	}
	texture.release();
	depths.release();
	normals.release();
	costs.release();
	hpm.CudaSpaceRelease(geom_consistency);
	hpm.ReleaseProblemHostMemory();
	std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void JointBilateralUpsampling(const std::string& dense_folder, const Problem& problem,
	int acmmp_size, ScaledImageCache& image_cache)
{
	std::stringstream result_path;
	result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
	std::string result_folder = result_path.str();
	std::string depth_path = result_folder + "/depths_geom.dmb";
	cv::Mat_<float> ref_depth;
	readDepthDmb(depth_path, ref_depth);

	std::vector<int> active_image_ids(1, problem.ref_image_id);
	image_cache.Retain(active_image_ids);
	const std::string scaled_path = ScaledImagePath(
		dense_folder, problem.ref_image_id, acmmp_size);
	cv::Mat scaled_color = image_cache.Get(problem.ref_image_id, scaled_path);
	cv::Mat scaled_gray, scaled_image_float;
	cv::cvtColor(scaled_color, scaled_gray, cv::COLOR_BGR2GRAY);
	scaled_gray.convertTo(scaled_image_float, CV_32FC1);

	std::cout << "Run JBU for image " << problem.ref_image_id << ".jpg" << std::endl;
	RunJBU(scaled_image_float, ref_depth, dense_folder, problem);
}

void RunFusion_Sky_Strict(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency)
{
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");
	std::string mask_folder = dense_folder + std::string("/masks");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> sky_masks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	sky_masks.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;

		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);

		std::stringstream sky_mask_path;
		sky_mask_path << mask_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> sky_mask = cv::imread(sky_mask_path.str(), cv::IMREAD_COLOR);

		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		cv::Mat_<cv::Vec3b>scaled_sky_mask;
		RescaleMask(sky_mask, scaled_sky_mask, depth);
		
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);

		images.push_back(scaled_image);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		masks.push_back(mask);
		sky_masks.push_back(scaled_sky_mask);
		image.release();
		sky_mask.release();
		depth.release();
		normal.release();
		scaled_image.release();
		mask.release();
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				bool view_strict = false;
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0)
					continue;

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				float consistent_Color[3] = { (float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2] };
				float segment_Color[3] = { (float)sky_masks[i].at<cv::Vec3b>(r, c)[0], (float)sky_masks[i].at<cv::Vec3b>(r, c)[1], (float)sky_masks[i].at<cv::Vec3b>(r, c)[2] };
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							float cons = exp(-tmp_index);
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				int view_num = 1;
				float factor = 0.3;


				if (num_consistent >= view_num && (dynamic_consistency > factor * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					
					float3 seg_Color = make_float3(segment_Color[0], segment_Color[1], segment_Color[2]);
					if ((int)seg_Color.x != 234 && (int)seg_Color.y != 235 && (int)seg_Color.z != 55) {
						PointCloud.push_back(point3D);
					}
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
					}
				}
			}
		}
	}

	std::string ply_path = dense_folder + "/HPM_MVS_plusplus/HPM_MVS_plusplus_mask.ply";
	ExportPointCloud(ply_path, PointCloud);
	ExportFusedPointCloud(dense_folder + "/HPM_MVS_plusplus/fused.ply", PointCloud);
}

void RunFusion(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency)
{
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());

		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.push_back(scaled_image);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.push_back(mask);


	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0)
					continue;

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				float consistent_Color[3] = { (float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2] };
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							float cons = exp(-tmp_index);
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					PointCloud.push_back(point3D);
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
					}
				}
			}
		}
	}

	std::string ply_path = dense_folder + "/HPM_MVS_plusplus/HPM_MVS_plusplus.ply";
	ExportPointCloud(ply_path, PointCloud);
	ExportFusedPointCloud(dense_folder + "/HPM_MVS_plusplus/fused.ply", PointCloud);
}

void ConfidenceEvaluation(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency) {
	size_t num_images = problems.size();
	std::string image_folder = dense_folder + std::string("/images");
	std::string cam_folder = dense_folder + std::string("/cams");

	std::vector<Camera> cameras;
	std::vector<cv::Mat_<float>> depths;
	std::vector<cv::Mat_<cv::Vec3f>> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat_<float>>consistency;
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	consistency.clear();
	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		std::stringstream image_path;
		image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
		cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
		std::stringstream cam_path;
		cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
		Camera camera = ReadCamera(cam_path.str());
		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string suffix = "/depths.dmb";
		if (geom_consistency) {
			suffix = "/depths_geom.dmb";
		}
		std::string depth_path = result_folder + suffix;
		std::string normal_path = result_folder + "/normals.dmb";
		cv::Mat_<float> depth;
		cv::Mat_<cv::Vec3f> normal;
		readDepthDmb(depth_path, depth);
		readNormalDmb(normal_path, normal);

		cv::Mat_<cv::Vec3b> scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		cameras.push_back(camera);
		depths.push_back(depth);
		normals.push_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.push_back(mask);
		cv::Mat consist = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
		consistency.push_back(consist);
		depth.release();
		normal.release();
		mask.release();
		consist.release();
		image.release();
		scaled_image.release();

	}
	for (size_t i = 0; i < num_images; ++i) {
		std::cout << "Hypothesis Confidence Evaluating Image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const int cols = depths[i].cols;
		const int rows = depths[i].rows;
		int num_ngb = problems[i].src_image_ids.size();
		std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (masks[i].at<uchar>(r, c) == 1)
					continue;
				float ref_depth = depths[i].at<float>(r, c);
				cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

				if (ref_depth <= 0.0) {
					continue;
				}
				if (ref_depth < cameras[i].depth_min || ref_depth > cameras[i].depth_max) {
					continue;
				}

				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
				float3 consistent_Point = PointX;
				cv::Vec3f consistent_normal = ref_normal;
				int num_consistent = 0;
				float dynamic_consistency = 0;

				for (int j = 0; j < num_ngb; ++j) {
					int src_id = problems[i].src_image_ids[j];
					const int src_cols = depths[src_id].cols;
					const int src_rows = depths[src_id].rows;
					float2 point;
					float proj_depth;
					ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_id].at<uchar>(src_r, src_c) == 1)
							continue;

						float src_depth = depths[src_id].at<float>(src_r, src_c);
						cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
						if (src_depth <= 0.0) {
							continue;
						}
						if (src_depth < cameras[i].depth_min || src_depth > cameras[i].depth_max) {
							continue;
						}

						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
						float2 tmp_pt;
						ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;

							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}

				if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
					consistency[i](r, c) = dynamic_consistency;

					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
						consistency[problems[i].src_image_ids[j]](used_list[j].y, used_list[j].x) = dynamic_consistency;
					}
				}
			}
		}
		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
		std::string result_folder = result_path.str();
		std::string mask_path = result_folder + "/confidence.dmb";
		writeDepthDmb(mask_path, consistency[i]);
	}
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	consistency.clear();

	cameras.shrink_to_fit();
	depths.shrink_to_fit();
	normals.shrink_to_fit();
	masks.shrink_to_fit();
	consistency.shrink_to_fit();
	std::cout << "Hypotheses Confidence Evaluating Over..." << std::endl;
}

void RemoveIntermediateResults(const std::string& dense_folder, const std::vector<Problem>& problems)
{
	size_t removed_files = 0;
	for (const Problem& problem : problems) {
		std::stringstream result_path;
		result_path << dense_folder << "/HPM_MVS_plusplus/2333_"
			<< std::setw(8) << std::setfill('0') << problem.ref_image_id;
		DIR* dir = opendir(result_path.str().c_str());
		if (dir == NULL) continue;

		struct dirent* entry;
		while ((entry = readdir(dir)) != NULL) {
			const std::string name(entry->d_name);
			const bool is_texture = name.compare(0, 7, "texture") == 0 &&
				name.size() >= 4 && name.compare(name.size() - 4, 4, ".dmb") == 0;
			const bool is_triangulation = name.compare(0, 13, "triangulation") == 0 &&
				name.size() >= 4 && name.compare(name.size() - 4, 4, ".png") == 0;
			const bool is_prior_depth = name.compare(0, 12, "depths_prior") == 0 &&
				name.size() >= 4 && name.compare(name.size() - 4, 4, ".dmb") == 0;
			const bool is_intermediate = name == "costs.dmb" || name == "confidence.dmb" ||
				is_texture || is_triangulation || is_prior_depth;
			if (is_intermediate) {
				const std::string file_path = result_path.str() + "/" + name;
				if (::remove(file_path.c_str()) == 0) ++removed_files;
			}
		}
		closedir(dir);
	}
	std::cout << "Removed " << removed_files
		<< " intermediate files; depth and normal maps were preserved." << std::endl;
}

int main(int argc, char** argv)
{
	const ExecutionTimer execution_timer;
	if (argc < 2) {
		std::cout << "USAGE: HPM-MVS_plusplus dense_folder [--initial-image-scale N] "
			"[--show-medium-result] [--mask] [--fusion-only/--fuse]\n"
			"  --initial-image-scale N: downscale width and height by N before pyramid processing (default 4; N=4 means 1/4 resolution)\n"
			"  --show-medium-result: keep intermediate cost, confidence, texture, triangulation, and prior debug files; depth/normal maps are always kept."
			<< std::endl;
		return -1;
	}

	std::string dense_folder = argv[1];
	bool mask_flag = false;
	bool fuse_only = false;
	bool show_medium_result = false;
	int initial_image_scale = 4;
	int arg = 2;
	try {
        while (arg < argc) {
            const std::string option(argv[arg++]);
            if (option == "--mask") mask_flag = true;
            else if (option == "--fusion-only" || option == "--fuse") fuse_only = true;
			else if (option == "--show-medium-result") show_medium_result = true;
			else if (option == "--initial-image-scale" && arg < argc) {
				const std::string value(argv[arg++]);
				size_t parsed = 0;
				initial_image_scale = std::stoi(value, &parsed);
				if (parsed != value.size())
					throw std::invalid_argument("--initial-image-scale must be a positive integer (1 = original resolution)");
				if (initial_image_scale < 1)
					throw std::invalid_argument("--initial-image-scale must be at least 1 (1 = original resolution)");
			}
            else throw std::invalid_argument("unknown or incomplete option: " + option);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

	std::vector<Problem> problems;
	GenerateSampleList(dense_folder, problems);
	std::cout << "Initial image scale: 1/" << initial_image_scale
		<< " (width and height are each scaled by 1/" << initial_image_scale << ")" << std::endl;
	std::cout << "Keep intermediate results: " << (show_medium_result ? "yes" : "no (removed after fusion)") << std::endl;

	std::string output_folder = dense_folder + std::string("/HPM_MVS_plusplus");
	// mkdir(output_folder.c_str());	//windows
	mkdir(output_folder.c_str(), 0755);	//linux

	size_t num_images = problems.size();
	std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

	if (fuse_only) {
		std::cout << "Fuse-only mode: skip MVS pipeline, run fusion with existing files." << std::endl;
		bool geom_consistency = true;
		if (mask_flag) {
			RunFusion_Sky_Strict(dense_folder, problems, geom_consistency);
		}
		else {
			RunFusion(dense_folder, problems, geom_consistency);
		}
		return 0;
	}

	int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems, initial_image_scale);
	size_t image_cache_capacity = 1;
	for (const Problem& problem : problems) {
		std::unordered_set<int> active_image_ids(problem.src_image_ids.begin(),
			problem.src_image_ids.end());
		active_image_ids.insert(problem.ref_image_id);
		image_cache_capacity = std::max(image_cache_capacity, active_image_ids.size());
	}
	if (image_cache_capacity > MAX_IMAGES) {
		std::cerr << "A ProcessProblem requires " << image_cache_capacity
			<< " images, exceeding MAX_IMAGES=" << MAX_IMAGES << "." << std::endl;
		return EXIT_FAILURE;
	}
	std::shared_ptr<ScaledImageCache> scaled_image_cache(
		new ScaledImageCache(image_cache_capacity));
	for (Problem& problem : problems) {
		problem.scaled_image_cache = scaled_image_cache;
	}

	int flag = 0;
	int geom_iterations;
	bool geom_consistency = false;
	bool prior_consistency = false;
	bool hierarchy = false;
	bool multi_geometry = false;
	bool mand_consistency = false;
	int max_hpm_scale = max_num_downscale;
	while (max_num_downscale >= 0) {
		geom_iterations = 3;
		std::cout << "Scale: " << max_num_downscale << std::endl;

		for (size_t i = 0; i < num_images; ++i) {
			if (problems[i].num_downscale >= 0) {
				problems[i].cur_max_side = problems[i].initial_max_side / pow(2, problems[i].num_downscale);
				problems[i].num_downscale--;
			}
		}
		const std::vector<std::string> scaled_image_files = PrepareScaleImages(
			dense_folder, problems, *scaled_image_cache);
		scaled_image_cache->ResetStats();

		if (flag == 0) {
			flag = 1;
			geom_consistency = false;
			prior_consistency = false;
			for (size_t i = 0; i < num_images; ++i) {
				ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, false, 0, show_medium_result);
			}
			prior_consistency = true;
			geom_consistency = false;
			for (int hpm_scale = max_hpm_scale; hpm_scale >= max_num_downscale; hpm_scale--) {
				ConfidenceEvaluation(dense_folder, problems, geom_consistency);
				for (size_t i = 0; i < num_images; ++i) {
					std::cout << "HPM Scale: " << hpm_scale << std::endl;
					int hpm_scale_distance = hpm_scale - problems[i].num_downscale - 1;
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, hpm_scale_distance, show_medium_result);
				}
			}

			geom_consistency = false;
			prior_consistency = false;
			for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
				if (geom_iter == 0) {
					multi_geometry = false;
				}
				else {
					multi_geometry = true;
				}

				if (geom_iter > 0) {
					mand_consistency = true;
					ConfidenceEvaluation(dense_folder, problems, geom_consistency);
					geom_consistency = true;
				}
				else {
					geom_consistency = true;
					mand_consistency = false;
				}

				for (size_t i = 0; i < num_images; ++i) {
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, 0, show_medium_result);
				}
			}
		}
		else {
			for (size_t i = 0; i < num_images; ++i) {
				JointBilateralUpsampling(dense_folder, problems[i],
					problems[i].cur_max_side, *scaled_image_cache);
			}

			hierarchy = true;
			mand_consistency = false;
			geom_consistency = false;
			prior_consistency = false;
			for (size_t i = 0; i < num_images; ++i) {
				ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, false, 0, show_medium_result);
			}
			hierarchy = false;
			prior_consistency = true;
			geom_consistency = false;
			multi_geometry = false;
			for (int hpm_scale = max_hpm_scale; hpm_scale >= max_num_downscale; hpm_scale--) {
				ConfidenceEvaluation(dense_folder, problems, geom_consistency);
				for (size_t i = 0; i < num_images; ++i) {
					std::cout << "HPM Scale: " << hpm_scale << std::endl;
					int hpm_scale_distance = hpm_scale - problems[i].num_downscale - 1;
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, hpm_scale_distance, show_medium_result);
				}
			}

			geom_consistency = false;
			prior_consistency = false;
			for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
				if (geom_iter == 0) {
					multi_geometry = false;
				}
				else {
					multi_geometry = true;
				}

				if (geom_iter > 0) {
					mand_consistency = true;
					ConfidenceEvaluation(dense_folder, problems, geom_consistency);
					geom_consistency = true;
				}
				else {
					geom_consistency = true;
					mand_consistency = false;
				}

				for (size_t i = 0; i < num_images; ++i) {
					ProcessProblem(dense_folder, problems, i, geom_consistency, prior_consistency, hierarchy, mand_consistency, max_num_downscale, multi_geometry, 0, show_medium_result);
				}
			}
		}
		std::cout << "Scaled-image cache for pyramid layer " << max_num_downscale
			<< ": " << scaled_image_cache->CacheHits() << " hits, "
			<< scaled_image_cache->CacheMisses() << " temporary-file reads, "
			<< scaled_image_cache->DiskReadSeconds() << " s reading cached images."
			<< std::endl;
		scaled_image_cache->Clear();
		RemoveScaleImageFiles(scaled_image_files);

		max_num_downscale--;
	}
	geom_consistency = true;
	if (mask_flag) {
		RunFusion_Sky_Strict(dense_folder, problems, geom_consistency);
	}
	else {
		RunFusion(dense_folder, problems, geom_consistency);
	}
	if (!show_medium_result) {
		RemoveIntermediateResults(dense_folder, problems);
	}
	return 0;
}
