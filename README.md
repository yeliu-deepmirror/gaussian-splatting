# 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Bernhard Kerbl*, Georgios Kopanas*, Thomas Leimkühler, George Drettakis (* indicates equal contribution)<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | [Video](https://youtu.be/T_kXY43VZnk) | [Other GRAPHDECO Publications](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH project page](https://fungraph.inria.fr) |<br>
| [T&T+DB COLMAP (650MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | [Pre-trained Models (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) | [Viewers for Windows (60MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) | [Evaluation Images (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) |<br>
![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models.

<a href="https://www.inria.fr/"><img style="width:20%;" src="assets/logo_inria.png"></a>
<a href="https://univ-cotedazur.eu/"><img style="width:20%;" src="assets/logo_uca.png"></a>
<a href="https://www.mpi-inf.mpg.de"><img style="width:20%;" src="assets/logo_mpi.png"></a>
<a href="https://team.inria.fr/graphdeco/"> <img style="width:20%;" src="assets/logo_graphdeco.png"></a>

Abstract: *Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (≥ 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.*


## Optimizer

See [the raw repo](https://github.com/graphdeco-inria/gaussian-splatting) for more instructions.
The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.

### Setup

**Fork to keep tracking the latest updates.**

#### Local Setup

create and run docker environment:
```shell
bash artifacts/docker/create_docker_image.sh
bash artifacts/docker/create_docker_container.sh
bash artifacts/docker/execute_docker_container.sh
```

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

### Prepare session

transform to colmap format:
```
bazel run -c opt //map/processor/output:colmap_proc_main -- \
-map_storage_output_directory=/Alpha/Data \
-map_storage_input_directories=/mnt/gz01/raw,/mnt/gz01/prod/spsg-cosplace-demos \
-session_name=${SESSION_NAME}
```

prepare lidar pointcloud data:
```
bazel run -c opt //map/tools:transform_pointcloud_main -- \
-map_storage_output_directory=/LidarMapping/data \
-map_storage_input_directories=/mnt/gz01/prod/spsg-cosplace-demos \
-session_name=${SESSION_NAME}
```

### Running

git clone project
```
git clone https://github.com/yeliu-deepmirror/gaussian-splatting --recursive
```

To run the optimizer, simply use :
```
python train.py --source_path ./Data/${SESSION_NAME}/colmap --resolution 1 --iterations 30_000
```

for general outdoor scenes (`--load_dynamic` if gpu memory not enough) :
```
# fast test
python train.py --source_path ./Data/${SESSION_NAME}/colmap --resolution 2 --iterations 30_000 \
--position_lr_init 0.000016 --scaling_lr 0.001 --front_only

# fine test
python train.py --source_path ./Data/${SESSION_NAME}/colmap --resolution 2 --iterations 30_000 --densify_until_iter 30_000 \
--position_lr_init 0.000016 --scaling_lr 0.001 --densify_grad_threshold 0.0002 --model_path output/ind

python train.py --source_path ./Data/${SESSION_NAME}/colmap --resolution 2 --iterations 120_000 --densify_until_iter 120_000 \
--position_lr_init 0.000016 --scaling_lr 0.001 --densify_grad_threshold 0.0002 --model_path output/ind
```


<br>

## Interactive Viewers
We provide two interactive viewers for our method: remote and real-time. Our viewing solutions are based on the [SIBR](https://sibr.gitlabpages.inria.fr/) framework, developed by the GRAPHDECO group for several novel-view synthesis projects.

#### build from source in Ubuntu 20.04
Backwards compatibility with Focal Fossa is not fully tested, but building SIBR with CMake should still work after invoking
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
sudo bash ./artifacts/docker/installers/opencv.sh
# Project setup
cd SIBR_viewers
git checkout fossa_compatibility
/gaussian-splatting/cmake/cmake-3.27.4-linux-x86_64/bin/cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
/gaussian-splatting/cmake/cmake-3.27.4-linux-x86_64/bin/cmake --build build -j8 --target install
```

### Navigation in SIBR Viewers
The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with ```W, A, S, D, Q, E``` for camera translation and ```I, K, J, L, U, O``` for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the ```Snap to``` button or find the closest camera with ```Snap to closest```. The floating menues also allow you to change the navigation speed. You can use the ```Scaling Modifier``` to control the size of the displayed Gaussians, or show the initial point cloud.

### Running the Real-Time Viewer

After extracting or installing the viewers, you may run the compiled ```SIBR_gaussianViewer_app[_config]``` app in ```<SIBR install dir>/bin```, e.g.:
```shell
TRAIN_ID=2fbbfa1f-1
/gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app \
-m /gaussian-splatting/output/${TRAIN_ID}
```

https://github.com/yeliu-deepmirror/gaussian-splatting/assets/74998488/cae483ec-e89c-418e-94f1-e8988f016fbd

https://github.com/yeliu-deepmirror/gaussian-splatting/assets/74998488/42c3a9d5-4476-4703-8b4c-c26ce74fd7a6

It should suffice to provide the ```-m``` parameter pointing to a trained model directory. Alternatively, you can specify an override location for training input data using ```-s```. To use a specific resolution other than the auto-chosen one, specify ```--rendering-size <width> <height>```. Combine it with ```--force-aspect-ratio``` if you want the exact resolution and don't mind image distortion.

**To unlock the full frame rate, please disable V-Sync on your machine and also in the application (Menu &rarr; Display). In a multi-GPU system (e.g., laptop) your OpenGL/Display GPU should be the same as your CUDA GPU (e.g., by setting the application's GPU preference on Windows, see below) for maximum performance.**

<!-- ![Teaser image](assets/select.png) -->

In addition to the initial point cloud and the splats, you also have the option to visualize the Gaussians by rendering them as ellipsoids from the floating menu.
SIBR has many other functionalities, please see the [documentation](https://sibr.gitlabpages.inria.fr/) for more details on the viewer, navigation options etc. There is also a Top View (available from the menu) that shows the placement of the input cameras and the original SfM point cloud; please note that Top View slows rendering when enabled. The real-time viewer also uses slightly more aggressive, fast culling, which can be toggled in the floating menu. If you ever encounter an issue that can be solved by turning fast culling off, please let us know.

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Real-Time Viewer</span></summary>

  #### --model-path / -m
  Path to trained model.
  #### --iteration
  Specifies which of state to load if multiple are available. Defaults to latest available iteration.
  #### --path / -s
  Argument to override model's path to source dataset.
  #### --rendering-size
  Takes two space separated numbers to define the resolution at which real-time rendering occurs, ```1200``` width by default. Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --load_images
  Flag to load source dataset images to be displayed in the top view for each camera.
  #### --device
  Index of CUDA device to use for rasterization if multiple are available, ```0``` by default.
  #### --no_interop
  Disables CUDA/GL interop forcibly. Use on systems that may not behave according to spec (e.g., WSL2 with MESA GL 4.5 software rendering).
</details>
<br>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
