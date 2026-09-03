# HPM-MVS++
* an enhanced version of HPM-MVS (HPM-MVS with Prior Consistency and Mandatory Consistency)
## News
* The code for [HPM-MVS](https://github.com/CLinvx/HPM-MVS) has been released.

## Dependencies
The code has been tested on Windows 10 with RTX 3070.<br />
 [cmake](https://cmake.org/)<br />
 [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 6.0<br />
 [OpenCV](https://opencv.org/) >= 2.4

`hnswlib` v0.8.0 is vendored under `third_party/hnswlib` (Apache-2.0), so no
separate nearest-neighbor package is required.

## HNSW planar prior

The planar-prior pass uses each reliable pixel's own plane hypothesis instead
of fitting a new plane from a Delaunay triangle. Reliable pixels are indexed by
their 2D image coordinates. Every other pixel queries its five nearest reliable
pixels and selects, among those five, the one with the smallest CIE Lab color
distance. The selected camera-frame plane is then used directly by planar-prior
consistency. Coarser support grids are mapped to the current full resolution;
their planes are not blended by joint bilateral upsampling.

The fixed HNSW parameters (`M=16`, `ef_construction=200`, `ef_search=64`) are
defined in `NearestPlanePrior.cpp`. Diagnostic outputs are written as
`depths_prior<scale>.dmb` and `hnsw_support_points<scale>.png`.

## Useage
* Compile
```
mkdir build
cd build
cmake ..
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to MVS input   
Run ./HPM-MVS_plusplus $data_folder true/flase(semantic segmentation masks for filtering sky area) to get reconstruction results 
```

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Ren_2023_ICCV,
    author    = {Ren, Chunlin and Xu, Qingshan and Zhang, Shikun and Yang, Jiaqi},
    title     = {Hierarchical Prior Mining for Non-local Multi-View Stereo},
    booktitle = {Proc. IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2023},
    pages     = {3611-3620}
}
```

## Acknowledgemets
This code largely benefits from the following repositories: [ACMH](https://github.com/GhiXu/ACMH), [ACMP](https://github.com/GhiXu/ACMP), [ACMMP](https://github.com/GhiXu/ACMMP). Thanks to their authors for opening source of their excellent works.
