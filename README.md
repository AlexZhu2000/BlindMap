
# BlindMap


## Main idea
**Abstract:** 

![Where2comm](./images/Intro.png)

## Features

- Dataset Support
  - [x] DAIR-V2X
  - [ ] OPV2V
  - [ ] V2X-Sim 2.0

- SOTA collaborative perception method support
    - [x] [Where2comm [Neurips2022]](https://arxiv.org/abs/2209.12836)
    - [x] [V2VNet [ECCV2020]](https://arxiv.org/abs/2008.07519)
    - [x] [DiscoNet [NeurIPS2021]](https://arxiv.org/abs/2111.00643)
    - [x] [V2X-ViT [ECCV2022]](https://arxiv.org/abs/2203.10638)
    - [x] [When2com [CVPR2020]](https://arxiv.org/abs/2006.00176)
    - [x] Late Fusion
    - [x] Early Fusion

- Visualization
  - [x] BEV visualization
  - [x] 3D visualization

## Citation


## Quick Start
### Install
Please refer to the [INSTALL.md](./docs/INSTALL.md) for detailed 
documentations. 

### Download dataset DAIR-V2X
1. Download raw data of [DAIR-V2X.](https://thudair.baai.ac.cn/cooptest)
2. Download complemented annotation from [Yifan Lu](https://github.com/yifanlu0227/CoAlign).


### Train your model
We adopt the same setting as OpenCOOD which uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --save_vis_n ${amount}
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', 'intermediate', 'no'(indicate no fusion, single agent), 'intermediate_with_comm'(adopt intermediate fusion and output the communication cost).
- `save_vis_n`: the amount of saving visualization result, default 10

The evaluation results  will be dumped in the model directory.

## Acknowledgements
Thank for the excellent cooperative perception codebases [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoPerception](https://github.com/coperception/coperception).

Thank for the excellent cooperative perception datasets [DAIR-V2X](https://thudair.baai.ac.cn/index), [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and [V2X-SIM](https://ai4ce.github.io/V2X-Sim/).

Thank for the dataset and code support by [YiFan Lu](https://github.com/yifanlu0227).

## Relevant Projects

Thanks for the insightful previous works in cooperative perception field.


**V2vnet: Vehicle-to-vehicle communication for joint perception and prediction** 
*ECCV20* [[Paper]](https://arxiv.org/abs/2008.07519) 

**When2com: Multi-agent perception via communication graph grouping** 
*CVPR20* [[Paper]](https://arxiv.org/abs/2006.00176) [[Code]](https://arxiv.org/abs/2006.00176)

**Who2com: Collaborative Perception via Learnable Handshake Communication** 
*ICRA20* [[Paper]](https://arxiv.org/abs/2003.09575?context=cs.RO)

**Learning Distilled Collaboration Graph for Multi-Agent Perception** 
*Neurips21* [[Paper]](https://arxiv.org/abs/2111.00643) [[Code]](https://github.com/DerrickXuNu/OpenCOOD)

**V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving** 
*RAL21* [[Paper]](https://arxiv.org/abs/2111.00643) [[Website]](https://ai4ce.github.io/V2X-Sim/)[[Code]](https://github.com/ai4ce/V2X-Sim)

**OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication** 
*ICRA2022* [[Paper]](https://arxiv.org/abs/2109.07644) [[Website]](https://mobility-lab.seas.ucla.edu/opv2v/) [[Code]](https://github.com/DerrickXuNu/OpenCOOD)

**V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer** *ECCV2022* [[Paper]](https://arxiv.org/abs/2203.10638) [[Code]](https://github.com/DerrickXuNu/v2x-vit) [[Talk]](https://course.zhidx.com/c/MmQ1YWUyMzM1M2I3YzVlZjE1NzM=)

**Self-Supervised Collaborative Scene Completion: Towards Task-Agnostic Multi-Robot Perception** 
*CoRL2022* [[Paper]](https://openreview.net/forum?id=hW0tcXOJas2)

**CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers** *CoRL2022* [[Paper]](https://arxiv.org/abs/2207.02202) [[Code]](https://github.com/DerrickXuNu/CoBEVT)

**DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection** *CVPR2022* [[Paper]](https://arxiv.org/abs/2204.05575) [[Website]](https://thudair.baai.ac.cn/index) [[Code]](https://github.com/AIR-THU/DAIR-V2X)


## Contact

If you have any problem with this code, please feel free to contact **XXXXX**.
