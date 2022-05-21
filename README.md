## TRI-VIDAR: TRI's Depth Estimation Repository

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/figs/tri-logo.png" width="25%"/>
</a>

[Installation](#installation) | [Configuration](#configuration) | [Datasets](#datasets) | [Visualization](#visualization) | [Publications](#publications) | [License](#license)

Official [PyTorch](https://pytorch.org/) repository for TRI's latest published depth estimation works. 
Our goal is to provide a clean environment to reproduce our results and facilitate further research in this field.
This repository is an updated version of [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm), our previous monocular depth estimation repository, featuring a different license. 


## Installation

We recommend using our provided dockerfile (see [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) instructions) to have a reproducible environment. 
To set up the repository, type in a terminal (only tested in Ubuntu 18.04):

```bash
git clone --recurse-submodules https://github.com/TRI-ML/vidar.git   # Clone repository with submodules
cd vidar                                                             # Move to repository folder
make docker-build                                                    # Build the docker image (recommended) 
```

To start our docker container, simply type `make docker-interactive`. From inside the docker, you can run scripts with the following command pattern:

```bash
python3 scripts/run.py <config.yaml>         # Single CPU/GPU  
python3 scripts/run_ddp.py <config.yaml>     # Distributed Data Parallel (DDP) multi-GPU
```

To verify that the environment is set up correctly, you can run a simple overfit test:

```bash
# Download a tiny subset of KITTI
curl -s https://tri-ml-public.s3.amazonaws.com/github/vidar/datasets/KITTI_tiny.tar | tar xv -C /data/datasets/
# Inside docker
python3 scripts/run.py configs/overfit/kitti_tiny.yaml
```

Once training is over (which takes around 1 minute), you should achieve results similar to this:

<img align="center" src="/media/figs/overfit_kitti.jpg" width="100%"/>

If you want to use features related to [AWS](https://aws.amazon.com/) (for dataset access) and [WandB](https://www.wandb.com/) (for experiment management), you can create associated accounts and configure your shell with the following environment variables:

```bash
export AWS_SECRET_ACCESS_KEY=something    # AWS secret key
export AWS_ACCESS_KEY_ID=something        # AWS access key
export AWS_DEFAULT_REGION=something       # AWS default region
export WANDB_ENTITY=something             # WANDB entity
export WANDB_API_KEY=something            # WANDB API key
```

## Configuration

Configuration files (stored in the `configs` folder) are the entry points for training and inference. 
The basic structure of a configuration file is:

```bash
wrapper:                            # Training parameters 
    <parameters>
arch:                               # Architecture used
    model:                          # Model file and parameters
        file: <model_file>
        <parameters>
    networks:                       # Networks available to the model
        network1:                   # Network1 file and parameters 
            file: <network1_file>
            <parameters>
        network2:                   # Network2 file and parameters 
            file: <network2_file>
            <parameters>
        ...
    losses:                         # Losses available to the model 
        loss1:                      # Loss1 file and parameters
            file: <loss1_file>      
            <parameters>
        loss2:                      # Loss2 file and parameters
            file: <loss2_file>      
            <parameters>
        ...        
evaluation:                         # Evaluation metrics for different tasks
    evaluation1:                    # Evaluation1 and parameters
        <parameters>
    evaluation2:                    # Evaluation2 and parameters
        <parameters>
    ...
optimizers:                         # Optimizers used to train the networks
    network1:                       # Optimizer for network1 and parameters
        <parameters>
    network2:                       # Optimizer for network2 and parameters
        <parameters>
    ...
datasets:                           # Datasets used
    train:                          # Training dataset and parameters
        <parameters>                
        augmentation:               # Training augmentations and parameters 
            <parameters>
        dataloader:                 # Training dataloader and parameters
            <parameters>
    validation:                     # Validation dataset and parameters
        <parameters>                
        augmentation:               # Validation augmentations and parameters
            <parameters>
        dataloader:                 # Validation dataloader and parameters
            <parameters>
```

To enable WandB logging, you can set these additional parameters in your configuration file:

```bash
wandb:
    folder: /data/vidar/wandb     # Where the wandb run is stored
    entity: your_entity           # Wandb entity
    project: your_project         # Wandb project
    num_validation_logs: X        # Number of visualization logs
    tags: [tag1,tag2,...]         # Wandb tags
    notes: note                   # Wandb notes
```

To enable checkpoint saving, you can set these additional parameters in your configuration file:

```bash
checkpoint:
    folder: /data/vidar/checkpoints       # Local folder to store checkpoints
    save_code: True                       # Save repository folder as well
    keep_top: 5                           # How many checkpoints should be stored
    s3_bucket: s3://path/to/s3/bucket     # [optional] AWS folder to store checkpoints        
    dataset: [0]                          # [optional] Validation dataset index to track
    monitor: [depth|abs_rel_pp_gt(0)_0]   # [optional] Validation metric to track
    mode: [min]                           # [optional] If the metric is minimized (min) or maximized (max)
```

To facilitate the reutilization of configuration files, we also provide a _recipe_ functionality, that enables parameter sharing. 
To use a recipe, simply type `recipe: <path/to/recipe>|<entry>` as an additional parameter, to copy all entries from that recipe onto that section. For example:

```bash
wrapper: 
  recipe: wrapper|default
```
will insert all parameters from section `default` of `configs/recipes/wrapper.yaml` onto the `wrapper` section of the configuration file. 
Parameters added after the recipe will overwrite those copied over, to facilitate customization.

## Datasets

In our provided configuration files, datasets are assumed to be downloaded in `/data/vidar/<dataset-name>`. 
For convenience, we provide links to some datasets we commonly use here (all licences still apply):

<table>
  <tr>
    <td style="text-align:center">Dataset</td>
    <td style="text-align:center">Version</td>
    <td style="text-align:center">Labels</td>
    <td style="text-align:center">Splits</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:left"><a href="http://www.cvlibs.net/datasets/kitti/">KITTI</a></td>
    <td style="text-align:center">KITTI_raw</td>
    <td style="text-align:center">RGB, Depth, Poses, Intrinsics</td>
    <td style="text-align:center">Train / Validation / Test</td>
  </tr>
  <tr>
    <td style="text-align:center">KITTI_tiny</td>
    <td style="text-align:center">RGB, Depth, Poses, Intrinsics</td>
    <td style="text-align:center">Train</td>
  </tr>
  <tr>
    <td rowspan="3" style="text-align:left"><a href="https://github.com/TRI-ML/DDAD">DDAD</a></td>
    <td style="text-align:center">DDAD_trainval</td>
    <td style="text-align:center">Depth prediction</td>
    <td style="text-align:center">Train / Validation</td>
  </tr>
  <tr>
    <td style="text-align:center">DDAD_tiny</td>
    <td style="text-align:center">Depth estimation</td>
    <td style="text-align:center">Train</td>
  </tr>
  <tr>
    <td style="text-align:center">DDAD_test</td>
    <td style="text-align:center">Depth estimation</td>
    <td style="text-align:center">Test</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:left"><a href="https://paralleldomain.com/public-datasets">PD</a></td>
    <td style="text-align:center">PD_guda</td>
    <td style="text-align:center">Depth prediction</td>
    <td style="text-align:center">Train / Validation</td>
  </tr>
  <tr>
    <td style="text-align:center">PD_draft</td>
    <td style="text-align:center">Depth estimation</td>
    <td style="text-align:center">Train / Validation</td>
  </tr>
  <tr>
    <td rowspan="2" style="text-align:left"><a href="https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/">VKITTI2</a></td>
    <td style="text-align:center">VKITTI2</td>
    <td style="text-align:center">Full Virtual KITTI 2 dataset</td>
    <td style="text-align:center">Train</td>
  </tr>
  <tr>
    <td style="text-align:center">VKITTI2_tiny</td>
    <td style="text-align:center">Tiny version of VKITTI2</td>
    <td style="text-align:center">Train</td>
  </tr>
</table>

## Visualization

We also provide tools for dataset and prediction visualization, based on our [CamViz](https://github.com/TRI-ML/camviz) library. 
It is added as a submodule in the `externals` folder. To use it from inside the docker, run `xhost +local:` before entering it. 
To visualize the information contained in different datasets, after it has been processed to be used by our repository, use the following command:

```bash
python3 demos/display_datasets/display_datasets.py <dataset>   
```

Some examples of visualization results you will generate for KITTI and DDAD are shown below (more examples can be found in the demo configuration file `demos/display_datasets/config.yaml`):

<img align="center" src="/media/figs/camviz_kitti.jpg" width="100%"/>
<img align="center" src="/media/figs/camviz_ddad.jpg" width="100%"/>

You can move the virtual viewing camera with the mouse, holding the left button to translate, the right button to rotate, and scrolling the wheel to zoom in/out. 
The up/down arrow keys change between temporal contexts, and the left/right arrow keys change between labels. 
Pressing SPACE changes between pointcloud color schemes (pixel color or per-camera). 

## Publications

### [3D Packing for Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1905.02693) (CVPR 2020, oral)
Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, Adrien Gaidon

**Abstract:** *Although cameras are ubiquitous, robotic platforms typically rely on active sensors like LiDAR for direct 3D perception. 
In this work, we propose a novel self-supervised monocular depth estimation method combining geometry with a new deep network, PackNet, learned only from unlabeled monocular videos. 
Our architecture leverages novel symmetrical packing and unpacking blocks to jointly learn to compress and decompress detail-preserving representations using 3D convolutions. 
Although self-supervised, our method outperforms other self, semi, and fully supervised methods on the KITTI benchmark. 
The 3D inductive bias in PackNet enables it to scale with input resolution and number of parameters without overfitting, generalizing better on out-of-domain data such as the NuScenes dataset. 
Furthermore, it does not require large-scale supervised pretraining on ImageNet and can run in real-time. 
Finally, we release DDAD (Dense Depth for Automated Driving), a new urban driving dataset with more challenging and accurate depth evaluation, thanks to longer-range and denser ground-truth depth generated from high-density LiDARs mounted on a fleet of self-driving cars operating world-wide.*

<p align="center">
  <img src="/media/figs/packnet.gif" width="50%"/>
</p>

<table>
  <tr>
    <td>GT depth</td>
    <td>Abs.Rel.</td>
    <td>Sq.Rel.</td>
    <td>RMSE</td>
    <td>RMSElog</td>
    <td>SILog</td>
    <td>d<sub>1.25</sub></td>
    <td>d<sub>1.25</sub><sup>2</sup></td>
    <td>d<sub>1.25</sub><sup>3</sup></td>
  </tr>
  <tr>
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/ResNet18_MR_selfsup_KITTI.ckpt"> ResNet18 | Self-Supervised | 192x640 | ImageNet &rightarrow; KITTI </a></td>
  </tr>
  <tr>
    <td style="text-align:left">Original</td>
    <td style="text-align:center">0.116</td>
    <td style="text-align:center">0.811</td>
    <td style="text-align:center">4.902</td>
    <td style="text-align:center">0.198</td>
    <td style="text-align:center">19.259</td>
    <td style="text-align:center">0.865</td>
    <td style="text-align:center">0.957</td>
    <td style="text-align:center">0.981</td>
  </tr>
  <tr>
    <td style="text-align:left">Improved</td>
    <td style="text-align:center">0.087</td>
    <td style="text-align:center">0.471</td>
    <td style="text-align:center">3.947</td>
    <td style="text-align:center">0.135</td>
    <td style="text-align:center">12.879</td>
    <td style="text-align:center">0.913</td>
    <td style="text-align:center">0.983</td>
    <td style="text-align:center">0.996</td>
  </tr>
  <tr>
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/PackNet_MR_selfsup_KITTI.ckpt****"> PackNet | Self-Supervised | 192x640 | KITTI </a></td>
  </tr>
  <tr>
    <td style="text-align:left">Original</td>
    <td style="text-align:center">0.111</td>
    <td style="text-align:center">0.800</td>
    <td style="text-align:center">4.576</td>
    <td style="text-align:center">0.189</td>
    <td style="text-align:center">18.504</td>
    <td style="text-align:center">0.880</td>
    <td style="text-align:center">0.960</td>
    <td style="text-align:center">0.982</td>
  </tr>
  <tr>
    <td style="text-align:left">Improved</td>
    <td style="text-align:center">0.078</td>
    <td style="text-align:center">0.420</td>
    <td style="text-align:center">3.485</td>
    <td style="text-align:center">0.121</td>
    <td style="text-align:center">11.725</td>
    <td style="text-align:center">0.931</td>
    <td style="text-align:center">0.986</td>
    <td style="text-align:center">0.996</td>
  </tr>
</table>

```
@inproceedings{tri-packnet,
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  author = {Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, Adrien Gaidon},
  booktitle = {Proceedings of the International Conference on Computer Vision and Pattern Recognition (CVPR)}
  year = {2020},
}
```

### [Multi-Frame Self-Supervised Depth Estimation with Transformers](https://arxiv.org/abs/2204.07616) (CVPR 2022)
Vitor Guizilini, Rares Ambrus, Dian Chen, Sergey Zakharov, Adrien Gaidon

**Abstract:** *Multi-frame depth estimation improves over single-frame approaches by also leveraging geometric relationships between images via feature matching, in addition to learning appearance-based features. 
In this paper we revisit feature matching for self-supervised monocular depth estimation, and propose a novel transformer architecture for cost volume generation. 
We use depth-discretized epipolar sampling to select matching candidates, and refine predictions through a series of self- and cross-attention layers. 
These layers sharpen the matching probability between pixel features, improving over standard similarity metrics prone to ambiguities and local minima. 
The refined cost volume is decoded into depth estimates, and the whole pipeline is trained end-to-end from videos using only a photometric objective. 
Experiments on the KITTI and DDAD datasets show that our DepthFormer architecture establishes a new state of the art in self-supervised monocular depth estimation, and is even competitive with highly specialized supervised single-frame architectures. 
We also show that our learned cross-attention network yields representations transferable across datasets, increasing the effectiveness of pre-training strategies.*

<p align="center">
  <img src="/media/figs/depthformer.gif" width="50%"/>
</p>

<table>
  <tr>
    <td>GT depth</td>
    <td>Frames</td>
    <td>Abs.Rel.</td>
    <td>Sq.Rel.</td>
    <td>RMSE</td>
    <td>RMSElog</td>
    <td>SILog</td>
    <td>d<sub>1.25</sub></td>
    <td>d<sub>1.25</sub><sup>2</sup></td>
    <td>d<sub>1.25</sub><sup>3</sup></td>
  </tr>
  <tr>
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/DepthFormer_MR_selfsup_KITTI.ckpt"> DepthFormer | Self-Supervised | 192x640 | ImageNet &rightarrow; KITTI </a></td>
  </tr>
  <tr>
    <td rowspan="2">Original</td>
    <td style="text-align:left">Single (t)</td>
    <td style="text-align:center">0.117</td>
    <td style="text-align:center">0.876</td>
    <td style="text-align:center">4.692</td>
    <td style="text-align:center">0.193</td>
    <td style="text-align:center">18.940</td>
    <td style="text-align:center">0.874</td>
    <td style="text-align:center">0.960</td>
    <td style="text-align:center">0.981</td>
  </tr>
  <tr>
    <td style="text-align:left">Multi (t-1,t)</td>
    <td style="text-align:center">0.090</td>
    <td style="text-align:center">0.661</td>
    <td style="text-align:center">4.149</td>
    <td style="text-align:center">0.175</td>
    <td style="text-align:center">17.260</td>
    <td style="text-align:center">0.905</td>
    <td style="text-align:center">0.963</td>
    <td style="text-align:center">0.982</td>
  </tr>
  <tr>
    <td rowspan="2">Improved</td>
    <td style="text-align:left">Single (t)</td>
    <td style="text-align:center">0.083</td>
    <td style="text-align:center">0.464</td>
    <td style="text-align:center">3.591</td>
    <td style="text-align:center">0.126</td>
    <td style="text-align:center">12.156</td>
    <td style="text-align:center">0.926</td>
    <td style="text-align:center">0.986</td>
    <td style="text-align:center">0.996</td>
  </tr>
  <tr>
    <td style="text-align:left">Multi (t-1,t)</td>
    <td style="text-align:center">0.055</td>
    <td style="text-align:center">0.271</td>
    <td style="text-align:center">2.917</td>
    <td style="text-align:center">0.095</td>
    <td style="text-align:center">9.160</td>
    <td style="text-align:center">0.955</td>
    <td style="text-align:center">0.991</td>
    <td style="text-align:center">0.998</td>
  </tr>
</table>

```
@inproceedings{tri-depthformer,
  title = {Multi-Frame Self-Supervised Depth with Transformers},
  author = {Vitor Guizilini, Rares Ambrus, Dian Chen, Sergey Zakharov, Adrien Gaidon},
  booktitle = {Proceedings of the International Conference on Computer Vision and Pattern Recognition (CVPR)}
  year = {2022},
}
``` 

### [Full Surround Monodepth from Multiple Cameras](https://arxiv.org/abs/2104.00152) (RA-L + ICRA 2022)
Vitor Guizilini, Igor Vasiljevic, Rares Ambrus, Greg Shakhnarovich, Adrien Gaidon

**Abstract:** *Self-supervised monocular depth and ego-motion estimation is a promising approach to replace or supplement expensive depth sensors such as LiDAR for robotics applications like autonomous driving. 
However, most research in this area focuses on a single monocular camera or stereo pairs that cover only a fraction of the scene around the vehicle. 
In this work, we extend monocular self-supervised depth and ego-motion estimation to large-baseline multi-camera rigs. 
Using generalized spatio-temporal contexts, pose consistency constraints, and carefully designed photometric loss masking, we learn a single network generating dense, consistent, and scale-aware point clouds that cover the same full surround 360 degree field of view as a typical LiDAR scanner. 
We also propose a new scale-consistent evaluation metric more suitable to multi-camera settings. 
Experiments on two challenging benchmarks illustrate the benefits of our approach over strong baselines.*

<p align="center">
  <img src="/media/figs/fsm.gif" width="50%"/>
</p>

<table>
  <tr>
    <td>Camera</td>
    <td>Abs.Rel.</td>
    <td>Sq.Rel.</td>
    <td>RMSE</td>
    <td>RMSElog</td>
    <td>SILog</td>
    <td>d<sub>1.25</sub></td>
    <td>d<sub>1.25</sub><sup>2</sup></td>
    <td>d<sub>1.25</sub><sup>3</sup></td>
  </tr>
  <tr>
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/FSM_MR_6cams_DDAD.ckpt"> FSM | Self-Supervised | 384x640 | ImageNet &rightarrow; DDAD </a></td>
  </tr>
  <tr>
    <td style="text-align:left">Front</td>
    <td style="text-align:center">0.131</td>
    <td style="text-align:center">2.940</td>
    <td style="text-align:center">14.252</td>
    <td style="text-align:center">0.237</td>
    <td style="text-align:center">22.226</td>
    <td style="text-align:center">0.824</td>
    <td style="text-align:center">0.935</td>
    <td style="text-align:center">0.969</td>
  </tr>
  <tr>
    <td style="text-align:left">Front Right</td>
    <td style="text-align:center">0.205</td>
    <td style="text-align:center">3.349</td>
    <td style="text-align:center">13.677</td>
    <td style="text-align:center">0.353</td>
    <td style="text-align:center">30.777</td>
    <td style="text-align:center">0.667</td>
    <td style="text-align:center">0.852</td>
    <td style="text-align:center">0.922</td>
  </tr>
  <tr>
    <td style="text-align:left">Back Right</td>
    <td style="text-align:center">0.243</td>
    <td style="text-align:center">3.493</td>
    <td style="text-align:center">12.266</td>
    <td style="text-align:center">0.394</td>
    <td style="text-align:center">33.842</td>
    <td style="text-align:center">0.594</td>
    <td style="text-align:center">0.821</td>
    <td style="text-align:center">0.907</td>
  </tr>
  <tr>
    <td style="text-align:left">Back</td>
    <td style="text-align:center">0.194</td>
    <td style="text-align:center">3.743</td>
    <td style="text-align:center">16.436</td>
    <td style="text-align:center">0.348</td>
    <td style="text-align:center">29.901</td>
    <td style="text-align:center">0.669</td>
    <td style="text-align:center">0.850</td>
    <td style="text-align:center">0.926</td>
  </tr>
  <tr>
    <td style="text-align:left">Back Left</td>
    <td style="text-align:center">0.235</td>
    <td style="text-align:center">3.641</td>
    <td style="text-align:center">13.570</td>
    <td style="text-align:center">0.387</td>
    <td style="text-align:center">31.765</td>
    <td style="text-align:center">0.594</td>
    <td style="text-align:center">0.816</td>
    <td style="text-align:center">0.907</td>
  </tr>
  <tr>
    <td style="text-align:left">Front Left</td>
    <td style="text-align:center">0.226</td>
    <td style="text-align:center">3.861</td>
    <td style="text-align:center">12.957</td>
    <td style="text-align:center">0.378</td>
    <td style="text-align:center">32.795</td>
    <td style="text-align:center">0.652</td>
    <td style="text-align:center">0.836</td>
    <td style="text-align:center">0.909</td>
  </tr>
</table>

```
@inproceedings{tri-fsm,
  title = {Full Surround Monodepth from Multiple Cameras},
  author = {Vitor Guizilini, Igor Vasiljevic, Rares Ambrus, Greg Shakhnarovich, Adrien Gaidon},
  booktitle = {Robotics and Automation Letters (RA-L)}
  year = {2022},
}
```

### [Self-Supervised Camera Self-Calibration from Videos](https://arxiv.org/abs/2112.03325) (ICRA 2022)
Jiading Fang, Igor Vasiljevic, Vitor Guizilini, Rares Ambrus, Greg Shakhnarovich, Adrien Gaidon, Matthew R.Walter

**Abstract:** *Camera calibration is integral to robotics and computer vision algorithms that seek to infer geometric properties of the scene from visual input streams. 
In practice, calibration is a laborious procedure requiring specialized data collection and careful tuning. 
This process must be repeated whenever the parameters of the camera change, which can be a frequent occurrence for mobile robots and autonomous vehicles. 
In contrast, self-supervised depth and ego-motion estimation approaches can bypass explicit calibration by inferring per-frame projection models that optimize a view synthesis objective. 
In this paper, we extend this approach to explicitly calibrate a wide range of cameras from raw videos in the wild. 
We propose a learning algorithm to regress per-sequence calibration parameters using an efficient family of general camera models. 
Our procedure achieves self-calibration results with sub-pixel reprojection error, outperforming other learning-based methods.
We validate our approach on a wide variety of camera geometries, including perspective, fisheye, and catadioptric. 
Finally, we show that our approach leads to improvements in the downstream task of depth estimation, achieving state-of-the-art results on the EuRoC dataset with greater computational efficiency than contemporary methods.*

<p align="center">
  <img src="/media/figs/self-calibration.gif" width="50%"/>
</p>

```
@inproceedings{tri-self_calibration,
  title = {Self-Supervised Camera Self-Calibration from Video},
  author = {Jiading Fang, Igor Vasiljevic, Vitor Guizilini, Rares Ambrus, Greg Shakhnarovich, Adrien Gaidon, Matthew Walter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}
  year = {2022},
}
```

## License

This repository is released under the [CC BY-NC 4.0](LICENSE.md) license.

