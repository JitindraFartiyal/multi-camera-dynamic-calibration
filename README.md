## Multi camera dynamic calibration

## Description
In this project, we evaluated different approaches to estimate the relative pose of multiple cameras. All the approaches facilitate dynamic addition or removal of camera without marker-based approaches.

Approaches used:
- Marker-based (For baseline)
- Keypoint-based
  - SIFT
  - SUPERGLUE
  - LOFTR
- Segmentation-based
  - FASTSAM+SSIM


## Installation
1. Clone this project

```
cd {your working directory} 
git clone https://gitos.rrze.fau.de/hex-teaching-2023/thesis/2023-fartiyal-jitindra.git
```
2. To install  all the libraries requirement
```
pip install -r requirements.txt
```
3. Fork or clone FASTSAM from and add to your project

```
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```



## Usage
1. To run marker-based approach (call the function from python)
```
get_marker_based_calibration(world_coord_filepath, calibration_dir, calibration_img_id1, calibration_img_id2)
```
2. To run keyppint-based approach (call the function from python)
```
run_non_marker_calibration(img_dir1, img_dir2, img_size, calibration_dir, keypoint_matching_algo)
```
keypoint_matching_algo = ["SIFT", "SUPERGLUE", "LOFTR"]

3. To run segmentation-based approach
```
run_segmentation_calibration(calibration_dir, img_dir1, img_dir2, similar_matches_path,
        segmented_output_path1, segmented_output_path2)
```
All the 3 approaches return average rotational similairty and average translational error

## Acknowledgements/References/Credits
- SUPERGLUE codebase was used from https://github.com/magicleap/SuperGluePretrainedNetwork.git
- FASTSAM codebase was used from https://github.com/CASIA-IVA-Lab/FastSAM.git
- For LOFTR we imported kornia library which is present in the requirement.txt