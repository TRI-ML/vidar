## TRI-VIDAR

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/figs/tri-logo.png" width="25%"/>
</a>

[Installation](#installation) | [Configuration](#configuration) | [Datasets](#datasets) | [Visualization](#visualization) | [Publications](#publications) | [License](#license)

Official [PyTorch](https://pytorch.org/) repository for some of TRI's latest publications, including self-supervised learning, multi-view geometry, and depth estimation. 
Our goal is to provide a clean environment to reproduce our results and facilitate further research in this field.
This repository is an updated version of [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm), our previous monocular depth estimation repository, featuring a different license. 

## Models

(Experimental) For convenient inference, we provide a growing list of our models (ZeroDepth, PackNet, DeFiNe) model over torchhub without installation.

### (New!) ZeroDepth
```python
import torch
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)

intrinsics = torch.tensor(np.load('examples/ddad_intrinsics.npy')).unsqueeze(0)
rgb = torch.tensor(imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.

depth_pred = zerodepth_model(rgb, intrinsics)
```

### PackNet
PackNet is a self-supervised monocular depth estimation model, to load a model trained on KITTI and run inference on an RGB image:
```python
import torch
packnet_model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True, trust_repo=True)
rgb = torch.tensor(imread('examples/ddad_sample.png')).permute(2,0,1).unsqueeze(0)/255.

depth_pred = model(rgb_image)
```

### DeFiNe
DeFiNe is a multi-view depth estimation model, to load a model trained on Scannet and run inference on multiple posed RGB images:
```python
import torch
define_model = torch.hub.load("TRI-ML/vidar", "DeFiNe", pretrained=True, trust_repo=True)
frames = {} 
frames["rgb"] = # a list of frames as 13HW torch.tensors
frames["intrinsics"] = # a list of 133 torch.tensor intrinsics matrices (one for each image)
frames["pose"] = # a batch of 144 relative poses to reference frame (one will be identity)
depth_preds = define_model(frames) # list of depths, one for each frame
```


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
python scripts/launch.py <config.yaml>         # Single CPU/GPU  
python scripts/launch_ddp.py <config.yaml>     # Distributed Data Parallel (DDP) multi-GPU
```

To verify that the environment is set up correctly, you can run a simple overfit test:

```bash
# Download a tiny subset of KITTI
mkdir /data/datasets 
curl -s https://tri-ml-public.s3.amazonaws.com/github/vidar/datasets/KITTI_tiny.tar | tar xv -C /data/datasets/
# Inside docker
python scripts/launch.py configs/overfit/kitti/selfsup_resnet18.yaml
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
    folder: /data/wandb     # Where the wandb run is stored
    entity: your_entity           # Wandb entity
    project: your_project         # Wandb project
    num_validation_logs: X        # Number of visualization logs
    tags: [tag1,tag2,...]         # Wandb tags
    notes: note                   # Wandb notes
```

To enable checkpoint saving, you can set these additional parameters in your configuration file:

```bash
checkpoint:
    folder: /data/checkpoints       # Local folder to store checkpoints
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

In our provided configuration files, datasets are assumed to be downloaded in `/data/datasets/<dataset-name>`. We have a separate repository for dataset management, that is a submodule of this repository and can be found [here](http://github.com/tri-ml/efm_datasets). It contains dataloaders for all datasets used in our works, as well as visualization tools that build upon our [CamViz](https://github.com/TRI-ML/camviz) library. 

```bash
cd externals/efm_datasets
python scripts/display_datasets/display_datasets.py scripts/config.yaml <dataset>   
```

Note that you need to execute `xhost +local:` before entering the docker with `make docker-interactive`, to enable local visualization. Some examples of visualization results you will generate for KITTI and DDAD are shown below:

<img align="center" src="/media/figs/camviz_kitti.jpg" width="100%"/>
<img align="center" src="/media/figs/camviz_ddad.jpg" width="100%"/>

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
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/PackNet_MR_selfsup_KITTI.ckpt"> PackNet | Self-Supervised | 192x640 | KITTI </a></td>
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

### [Depth Field Networks for Generalizable Multi-view Scene Representation](https://arxiv.org/abs/2207.14287) (ECCV 2022)
Vitor Guizilini, Igor Vasiljevic, Jiading Fang, Rares Ambrus, Greg Shakhnarovich, Matthew Walter, Adrien Gaidon

**Abstract:** *Modern 3D computer vision leverages learning to boost geometric reasoning, mapping image data to classical structures such as cost volumes or epipolar constraints to improve matching. These architectures are specialized according to the particular problem, and thus require significant task-specific tuning, often leading to poor domain generalization performance. Recently, generalist Transformer architectures have achieved impressive results in tasks such as optical flow and depth estimation by encoding geometric priors as inputs rather than as enforced constraints. In this paper, we extend this idea and propose to learn an implicit, multi-view consistent scene representation, introducing a series of 3D data augmentation techniques as a geometric inductive prior to increase view diversity. We also show that introducing view synthesis as an auxiliary task further improves depth estimation. Our Depth Field Networks (DeFiNe) achieve state-of-the-art results in stereo and video depth estimation without explicit geometric constraints, and improve on zero-shot domain generalization by a wide margin.*

<p align="center">
  <img src="/media/figs/define.gif" width="50%"/>
</p>

```
@inproceedings{tri-define,
  title={Depth Field Networks For Generalizable Multi-view Scene Representation},
  author={Guizilini, Vitor and Vasiljevic, Igor and Fang, Jiading and Ambrus, Rares and Shakhnarovich, Greg and Walter, Matthew R and Gaidon, Adrien},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXII},
  pages={245--262},
  year={2022},
  organization={Springer}
}
```

### [Towards Zero-Shot Scale-Aware Monocular Depth Estimation](https://arxiv.org/abs/2306.17253) (ICCV 2023)
Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares Ambrus, Adrien Gaidon

**Abstract:** *Monocular depth estimation is scale-ambiguous, and thus requires scale supervision to produce metric predictions. Even so, the resulting models will be geometry-specific, with learned scales that cannot be directly transferred across domains. Because of that, recent works focus instead on relative depth, eschewing scale in favor of improved up-to-scale zero-shot transfer. In this work we introduce ZeroDepth, a novel monocular depth estimation framework capable of predicting metric scale for arbitrary test images from different domains and camera parameters. This is achieved by (i) the use of input-level geometric embeddings that enable the network to learn a scale prior over objects; and (ii) decoupling the encoder and decoder stages, via a variational latent representation that is conditioned on single frame information. We evaluated ZeroDepth targeting both outdoor (KITTI, DDAD, nuScenes) and indoor (NYUv2) benchmarks, and achieved a new state-of-the-art in both settings using the same pre-trained model, outperforming methods that train on in-domain data and require test-time scaling to produce metric estimates.*

<p align="center">
  <img src="/media/figs/zerodepth.gif" width="50%"/>
</p>

```
@inproceedings{tri-zerodepth,
  title={Towards Zero-Shot Scale-Aware Monocular Depth Estimation},
  author={Guizilini, Vitor and Vasiljevic, Igor and Chen, Dian and Ambrus, Rares and Gaidon, Adrien},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month={October},
  year={2023},
}
```

### [Robust Self-Supervised Extrinsic Self-Calibration](https://arxiv.org/pdf/2308.02153.pdf) (IROS 2023)
Takayuki Kanai, Igor Vasiljevic, Vitor Guizilini, Adrien Gaidon, Rares Ambrus

**Abstract:** *Autonomous vehicles and robots need to operate over a wide variety of scenarios in order to complete tasks efficiently and safely. Multi-camera self-supervised monocular depth estimation from videos is a promising way to reason about the environment, as it generates metrically scaled geometric predictions from visual data without requiring additional sensors. However, most works assume well-calibrated extrinsics to fully leverage this multi-camera setup, even though accurate and efficient calibration is still a challenging problem. In this work, we introduce a novel method for extrinsic calibration that builds upon the principles of self-supervised monocular depth and ego-motion learning. Our proposed curriculum learning strategy uses monocular depth and pose estimators with velocity supervision to estimate extrinsics, and then jointly learns extrinsic calibration along with depth and pose for a set of overlapping cameras rigidly attached to a moving vehicle. Experiments on a benchmark multi-camera dataset (DDAD) demonstrate that our method enables self-calibration in various scenes robustly and efficiently compared to a traditional vision-based pose estimation pipeline. Furthermore, we demonstrate the benefits of extrinsics self-calibration as a way to improve depth prediction via joint optimization.*

<p align="center">
  <img src="/media/figs/sesc_teaser.gif" width="50%"/>
</p>

<table>
  <tr>
    <td>Abs.Rel.</td>
    <td>Front</td>
    <td>F.Left</td>
    <td>F.Right</td>
    <td>B.Left</td>
    <td>B.Right</td>
    <td>Back</td>
  </tr>
  <tr>
    <td colspan="9"><a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/models/SESC_MR_withExtrinsics_DDAD.ckpt"> SESC | Self-Supervised | 384x640 | ImageNet &rightarrow; DDAD </a> (<a href="https://tri-ml-public.s3.amazonaws.com/github/vidar/data/extrinsics-ddad-everything.pkl">GT-extrinsics</a>)</td>
  </tr>
  <tr>
    <td style="text-align:left">Metric (PP)</td>
    <td style="text-align:center">0.159</td>
    <td style="text-align:center">0.194</td>
    <td style="text-align:center">0.213</td>
    <td style="text-align:center">0.212</td>
    <td style="text-align:center">0.220</td>
    <td style="text-align:center">0.205</td>
  </tr>
  <tr>
    <td style="text-align:left">Scaled (PP_MD)</td>
    <td style="text-align:center">0.156</td>
    <td style="text-align:center">0.200</td>
    <td style="text-align:center">0.225</td>
    <td style="text-align:center">0.217</td>
    <td style="text-align:center">0.234</td>
    <td style="text-align:center">0.216</td>
  </tr>
</table>

```
@inproceedings{tri_sesc_iros23,
  title = {Robust Self-Supervised Extrinsic Self-Calibration},
  author = {Takayuki Kanai and Igor Vasiljevic and Vitor Guizilini and Adrien Gaidon and Rares Ambrus},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2023},
}
```

### [Self-Supervised Geometry-Guided Initialization for Robust Monocular Visual Odometry](https://arxiv.org/abs/2406.00929v1) (arXiv 2024)
Takayuki Kanai, Igor Vasiljevic, Vitor Guizilini, Kazuhiro Shintani

**Abstract:** *Monocular visual odometry is a key technology in a wide variety of autonomous systems. Traditional feature-based methods suffer from failures due to poor lighting, insufficient texture, large motions, etc. In contrast, recent learning-based dense SLAM methods exploit iterative dense bundle adjustment to address such failure cases, and achieve robust and accurate localization in a wide variety of real environments, without depending on domain-specific training data. However, despite its potential, the method still struggles with scenarios involving large motion, object dynamics, etc. In this paper, we diagnose key weaknesses in a popular learning-based dense SLAM model (DROID-SLAM) by analyzing major failure cases on outdoor benchmarks and exposing various shortcomings of its optimization process. We then propose the use of self-supervised priors leveraging a frozen large-scale pre-trained monocular depth estimation to initialize the dense bundle adjustment process, leading to robust visual odometry without the need to fine-tune the SLAM backbone. Despite its simplicity, our proposed method demonstrates significant improvements on KITTI odometry, as well as the challenging DDAD benchmark.* 	Project page: [this https URL](https://toyotafrc.github.io/SGInit-Proj/).

<p align="center">
  <img src="/media/figs/sginit.gif" width="60%"/>
</p>


```
@article{frc-tri-sginit,
        title={Self-Supervised Geometry-Guided Initialization for Robust Monocular Visual Odometry}, 
        author={Takayuki Kanai and Igor Vasiljevic and Vitor Guizilini and Kazuhiro Shintani},
        year={2024},
        journal={arXiv},
}
```

## License

This repository is released under the [CC BY-NC 4.0](LICENSE.md) license.

