<h1 align = "center">
  Pytorch Deep Learning Models <br>
  <a href="https://github.com/m-np/pytorch-dl-models/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/m-np/pytorch-dl-models?logo=git&style=plastic"></a>
  <a href="https://github.com/m-np/pytorch-dl-models/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/m-np/pytorch-dl-models?style=plastic&logo=github"></a>
  <a href="https://github.com/m-np/pytorch-dl-models/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/m-np/pytorch-dl-models?style=plastic&logo=github"></a>
  <a href="https://makeapullrequest.com/"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=plastic&logo=open-source-initiative"></a>
</h1>

<div align = "justify">

**Objective:** This repo is designed to build different Deep Learning models from scratch in Pytorch. Here, we are focused on understanding the different building blocks that make up different models. </br>
Once the models are developed they can be trained on different datasets however I have not linked different datasets in this repo. </br>
To understand the repo check [**HOWTO.md**](./HOWTO.md) file.

---

</div>

## Setup

Please follow the following steps to run the project locally <br/>

1. `git clone https://github.com/m-np/pytorch-dl-models.git`
2. Open Anaconda console/Terminal and navigate into project directory `cd path_to_repo`
3. Run `conda create --name <env_name> python==3.9`.
4. Run `conda activate <env_name>` (for running scripts from your console or set the interpreter in your IDE)

For adding the new conda environment to the jupyter notebook follow this additional instruction
1. Run `conda install -c anaconda ipykernel`
2. Run `python -m ipykernel install --user --name=<env_name>`

-----

For pytorch installation:

PyTorch pip package will come bundled with some version of CUDA/cuDNN with it,
but it is highly recommended that you install a system-wide CUDA beforehand, mostly because of the GPU drivers. 
I also recommend using Miniconda installer to get conda on your system.
Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md)
and use the most up-to-date versions of Miniconda and CUDA/cuDNN for your system.

-----

For other module installation, please follow the following steps:
1. Open Anaconda console/Terminal and navigate into project directory `cd path_to_repo`
2. Run `conda activate <env_name>`
3. Run `pip install -r requirements.txt` found 👉 [`requirements.txt`](./requirements.txt)

## Table of content

This repo showcases models from the below ML tasks

| ML Task    | ML Model | Description |
| --------- | ------- | ------ |
| CV     | [LeNet](src/models/cv/lenet) | Designed for Image Classification Task |
|        | [AlexNet](src/models/cv/alexnet) | Designed for Image Classification Task |
|        | [VGG16](src/models/cv/vggnet) | Designed for Image Classification Task |
|        | [ResNet50](src/models/cv/resnet) | Designed for Image Classification Task |
|        | [InceptionV1](src/models/cv/inception) | Designed for Image Classification Task |
|        | [MobilenetV1](src/models/cv/mobilenet) | Designed for Image Classification Task |
| NLP    | [Transformer](src/models/nlp/original_transformer) | Designed for Machine Translation |
|        | [Bert](src/models/nlp/Bert) | Designed for Language Modeling |
| Audio  | [Wavenet](src/models/audio/Wavenet) | Designed for Speech generation |
|        | [Tacotron2](src/models/audio/Tacotron2) | Designed for Speech |
| Recommendation System  | [Ripplenet](src/models/recommendation_system/ripplenet) | Designed for KG style personalization model |

The above models are also registered in the following [model registry](src/model_registry.py)

## Usage

For visualizing a model for a task, please follow the following steps after running `conda activate <env_name>` 

```python
python main.py -t <MLTask> -m <MLModel as per above table>
```
You can also check the various MLTask/MLModel in the following [model registry](src/model_registry.py)

## LICENSE 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


## Resources

<p align = "justify">
  References obtained from following sources :
   
AlexNet - [blog](https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5)  </br>
ResNet50 - [blog](https://medium.com/@freshtechyy/a-detailed-introduction-to-resnet-and-its-implementation-in-pytorch-744b13c8074a) </br>
InceptionV1 - [research paper](https://arxiv.org/abs/1409.4842) </br>
MobilenetV1 - [research paper](https://arxiv.org/abs/1704.04861) </br>
Transformer - [blog](https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a) [research paper](https://arxiv.org/abs/1706.03762) </br>
Bert - [blog](https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a) [research paper](https://arxiv.org/abs/1810.04805) </br>
Wavenet - [research paper](https://arxiv.org/abs/1609.03499) [reference code](https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/modules.py) </br>
Tacotron2 - [research paper](https://arxiv.org/abs/1712.05884) [reference code](https://github.com/NVIDIA/tacotron2/blob/master/model.py) </br>
</p>
