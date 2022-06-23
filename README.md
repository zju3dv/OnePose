# OnePose: One-Shot Object Pose Estimation without CAD Models
### [Project Page](https://zju3dv.github.io/onepose) | [Paper](https://arxiv.org/pdf/2205.12257.pdf)
<br/>

> OnePose: One-Shot Object Pose Estimation without CAD Models  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Zihao Wang](http://zihaowang.xyz/)<sup>\*</sup>, [Siyu Zhang](https://derizsy.github.io/)<sup>\*</sup>, [Xingyi He](https://github.com/hxy-123/), [Hongcheng Zhao](https://github.com/HongchengZhao), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Xiaowei Zhou](https://xzhou.me)   
> CVPR 2022  

![demo_vid](assets/onepose-github-teaser.gif)

## TODO List
- [x] Training and inference code.
- [x] Pipeline to reproduce the evaluation results on the proposed OnePose dataset.
- [ ] `OnePose Cap` app: we are preparing for the release of the data capture app to the App Store (iOS only), please stay tuned.
- [ ] Demo pipeline for running OnePose with custom-captured data including the online tracking module.

## Installation

```shell
conda env create -f environment.yaml
conda activate onepose
```
We use [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperPointPretrainedNetwork) 
for 2D feature detection and matching in this project.
We can't provide the code directly due its LICENSE requirements, please download the inference code and pretrained models using the following script：
```shell
REPO_ROOT=/path/to/OnePose
cd $REPO_ROOT
sh ./scripts/prepare_2D_matching_resources.sh
```

[COLMAP](https://colmap.github.io/) is used in this project for Structure-from-Motion. 
Please refer to the official [instructions](https://colmap.github.io/install.html) for the installation.

[Optional, WIP] You may optionally try out our web-based 3D visualization tool [Wis3D](https://github.com/zju3dv/Wis3D) for convenient and interactive visualizations of feature matches. We also provide many other cool visualization features in Wis3D, welcome to try it out.

```bash
# Working in progress, should be ready very soon, only available on test-pypi now.
pip install -i https://test.pypi.org/simple/ wis3d
```

## Training and Evaluation on OnePose dataset
### Dataset setup 
1. Download OnePose dataset from [onedrive storage](https://zjueducn-my.sharepoint.com/:f:/g/personal/zihaowang_zju_edu_cn/ElfzHE0sTXxNndx6uDLWlbYB-2zWuLfjNr56WxF11_DwSg?e=GKI0Df) and extract them into `$/your/path/to/onepose_datasets`. 
The directory should be organized in the following structure:
    ```
    |--- /your/path/to/onepose_datasets
    |       |--- train_data
    |       |--- val_data
    |       |--- test_data
    |       |--- sample_data
    ```

2. Build the dataset symlinks
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /your/path/to/onepose_datasets $REPO_ROOT/data/onepose_datasets
    ```

3. Run Structure-from-Motion for the data sequences

    Reconstructed the object point cloud and 2D-3D correspondences are needed for both training and test objects:
    ```python
    python run.py +preprocess=sfm_spp_spg_train.yaml # for training data
    python run.py +preprocess=sfm_spp_spg_test.yaml # for testing data
    python run.py +preprocess=sfm_spp_spg_val.yaml # for val data
    python run.py +preprocess=sfm_spp_spg_sample.yaml # an example, if you don't want to test the full dataset
    ```

### Inference on OnePose dataset
1. Download the pretrain weights [pretrained model](https://drive.google.com/drive/folders/1VjLLjJ9oxjKV5Xy3Aty0uQUVwyEhgtIE?usp=sharing) and move it to `${REPO_ROOT}/data/model/checkpoints/onepose/GATsSPG.ckpt`.

2. Inference with category-agnostic 2D object detection.

    When deploying OnePose to a real world system, 
    an off-the-shelf category-level 2D object detector like [YOLOv5](https://github.com/ultralytics/yolov5) can be used.
    However, this could defeat the category-agnostic nature of OnePose.
    We can instead use a feature-matching-based pipeline for 2D object detection, which locates the scanned object on the query image through 2D feature matching.
    Note that the 2D object detection is only necessary during the initialization.
    After the initialization, the 2D bounding box can be obtained from projecting the previously detected 3D bounding box to the current camera frame.
    Please refer to the [supplementary material](https://zju3dv.github.io/onepose/files/onepose_supp.pdf) for more details. 

    ```python
    # Obtaining category-agnostic 2D object detection results first.
    # Increasing the `n_ref_view` will improve the detection robustness but with the cost of slowing down the initialization speed.
    python feature_matching_object_detector.py +experiment=object_detector.yaml n_ref_view=15

    # Running pose estimation with `object_detect_mode` set to `feature_matching`.
    # Note that enabling visualization will slow down the inference.
    python inference.py +experiment=test_GATsSPG.yaml object_detect_mode=feature_matching save_wis3d=False
    ```

3. Running inference with ground-truth 2D bounding boxes

    The following command should reproduce results in the paper, which use 2D boxes projected from 3D boxes as object detection results.

    ```python
    # Note that enabling visualization will slow down the inference.
    python inference.py +experiment=test_GATsSPG.yaml object_detect_mode=GT_box save_wis3d=False # for testing data
    ```


    
4. [Optional] Visualize matching and estimated poses with Wis3D. Make sure the flag `save_wis3d` is set as True in testing 
and the full images are extracted from `Frames.m4v` by script `scripts/parse_full_img.sh`. 
The visualization file will be saved under `cfg.output.vis_dir` directory which is set as `GATsSPG` by default. 
Run the following commands for visualization:
    ```shell
    sh ./scripts/parse_full_img.sh path_to_Frames_m4v # parse full image from m4v file
    
    cd runs/vis/GATsSPG
    wis3d --vis_dir ./ --host localhost --port 11020
    ```
    This would launch a web service for visualization at port 11020.


### Training the GATs Network
1. Prepare ground-truth annotations. Merge annotations of training/val data:
    ```python
    python run.py +preprocess=merge_anno task_name=onepose split=train
    python run.py +preprocess=merge_anno task_name=onepose split=val
    ```
   
2. Begin training
    ```python
    python train.py +experiment=train_GATsSPG task_name=onepose exp_name=training_onepose
    ```
   
All model weights will be saved under `${REPO_ROOT}/data/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/data/logs/${exp_name}`.
<!-- You can visualize the training process by tensorboard:
```shell
tensorboard xx
``` -->

## Citation
If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{sun2022onepose,
	title={{OnePose}: One-Shot Object Pose Estimation without {CAD} Models},
	author = {Sun, Jiaming and Wang, Zihao and Zhang, Siyu and He, Xingyi and Zhao, Hongcheng and Zhang, Guofeng and Zhou, Xiaowei},
	journal={CVPR},
	year={2022},
}
```

## Copyright

This work is affiliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright SenseTime. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgement
Part of our code is borrowed from [hloc](https://github.com/cvg/Hierarchical-Localization) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), thanks to their authors for the great works.