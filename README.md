## *BiPointNet: Binary Neural Network for Point Clouds*

Created by [Haotong Qin](https://htqin.github.io/), [Zhongang Cai](https://scholar.google.com/citations?user=WrDKqIAAAAAJ&hl=en), [Mingyuan Zhang](https://scholar.google.com/citations?user=2QLD4fAAAAAJ&hl=en), Yifu Ding, Haiyu Zhao, Shuai Yi, [Xianglong Liu](http://sites.nlsde.buaa.edu.cn/~xlliu/), and [Hao Su](https://cseweb.ucsd.edu/~haosu/) from Beihang University, SenseTime, and UCSD.

![prediction example](https://htqin.github.io/Imgs/ICLR/overview_v1.png)

### Introduction

This project is the official implementation of our accepted ICLR 2021 paper *BiPointNet: Binary Neural Network for Point Clouds* [[PDF]( https://openreview.net/forum?id=9QLRCVysdlO)]. To alleviate the resource constraint for real-time point cloud applications that run on edge devices, in this paper we present ***BiPointNet***, the first model binarization approach for efficient deep learning on point clouds. We first discover that the immense performance drop of binarized models for point clouds mainly stems from two challenges: aggregation-induced feature homogenization that leads to a degradation of information entropy, and scale distortion that hinders optimization and invalidates scale-sensitive structures. With theoretical justifications and in-depth analysis, our BiPointNet introduces Entropy-Maximizing Aggregation (EMA) to modulate the distribution before aggregation for the maximum information entropy, and Layer-wise Scale Recovery (LSR) to efficiently restore feature representation capacity. Extensive experiments show that BiPointNet outperforms existing binarization methods by convincing margins, at the level even comparable with the full precision counterpart. We highlight that our techniques are generic, guaranteeing significant improvements on various fundamental tasks and mainstream backbones, e.g., BiPointNet gives an impressive 14.7x speedup and 18.9x storage saving on real-world resource-constrained devices.

### Installation
```shell script
# create new conda environment
conda create -n pyg python=3.7 -y
conda activate pyg

# install pytorch
conda install pytorch==1.5.0 torchvision cudatoolkit=10.1 -c pytorch -y

# install pytorch-geometric
export CUDA=cu101
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric

# install other dependencies
pip install pyyaml
```

### Training

```shell script
export PYTHONPATH=$(pwd):$PYTHONPATH
conda activate pyg
python scripts/main.py ${CONFIG} ${PYTHON_ARGS}
```

### Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{Qin:iclr21,
  author    = {Haotong Qin and Zhongang Cai and Mingyuan Zhang 
  and Yifu Ding and Haiyu Zhao and Shuai Yi 
  and Xianglong Liu and Hao Su},
  title     = {BiPointNet: Binary Neural Network for Point Clouds},
  booktitle = {ICLR},
  year      = {2021}
}
```
