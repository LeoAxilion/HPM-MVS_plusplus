# HPM-MVS++
* an enhanced version of HPM-MVS (HPM-MVS with Prior Consistency and Mandatory Consistency)
## News
* The code for [HPM-MVS](https://github.com/CLinvx/HPM-MVS) has been released.

## Dependencies
The code has been tested on Windows 10 with RTX 3070.<br />
 [cmake](https://cmake.org/)<br />
 [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 6.0<br />
 [OpenCV](https://opencv.org/) >= 2.4

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
Run `./HPM-MVS_plusplus $data_folder` to get reconstruction results.
```

### Runtime options

`--initial-image-scale N` sets the initial downscale divisor for HPM-MVS++ images. The default is `4`: both width and height are scaled to one quarter of their input dimensions before the internal pyramid processing. Use `1` to keep the original resolution. For example:

```
./HPM-MVS_plusplus "$data_folder" --initial-image-scale 4
```

Intermediate files are removed after successful fusion by default. Add `--show-medium-result` to keep intermediate cost, confidence, texture, triangulation, and planar-prior debug files. The per-image depth maps (`depths.dmb` and `depths_geom.dmb`) and normal maps (`normals.dmb`) are retained regardless of this option:

```
./HPM-MVS_plusplus "$data_folder" --initial-image-scale 4 --show-medium-result
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
