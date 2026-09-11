# BiPointNet: Binary Neural Network for Point Clouds

**ICLR 2021**

Haotong Qin, Zhongang Cai, Mingyuan Zhang, Yifu Ding, Haiyu Zhao, Shuai Yi, Xianglong Liu, Hao Su

[Paper](https://openreview.net/forum?id=9QLRCVysdlO) | [arXiv](https://arxiv.org/abs/2010.05501) | [Citation](#citation)

**BiPointNet binarizes point-cloud networks while preserving information through aggregation and restoring feature scale.** Entropy-Maximizing Aggregation (EMA) addresses feature homogenization; Layer-wise Scale Recovery (LSR) corrects scale distortion. It requires training on point-cloud data.

## Published results

ModelNet40 classification, overall accuracy (%) from Table 3. W/A denotes weights/activations; the paper retains selected sensitive layers at full precision. Baseline XNOR uses the original aggregation; BiPointNet uses EMA-max in these rows.

| Backbone | Full precision (32/32) | XNOR (1/1) | BiPointNet (1/1) |
| --- | --- | --- | --- |
| PointNet (vanilla) | 86.8 | 61.0 | 85.6 |
| PointNet | 88.2 | 64.9 | 86.4 |
| PointNet++ | 90.0 | 63.1 | 87.8 |
| DGCNN | 89.2 | 51.5 | 83.4 |

The paper reports **14.7× measured speedup on ARM Cortex-A72 and 18.9× parameter-storage saving** for its deployment configuration (Section 4.3; Appendix B, Table 4). The A72 device is Raspberry Pi 4B (1.5 GHz); Raspberry Pi 3B with Cortex-A53 is evaluated separately. These figures use the paper's optimized binary implementation and are not promised speedups from ordinary PyTorch tensor operations.

### What this paper supports

- Aggregation can destroy information in binarized point features; EMA explicitly targets this bottleneck (Section 3.2).
- LSR restores feature scale with layer-wise factors (Section 3.3; Tables 1–2).
- PointNet ModelNet40 accuracy reaches 86.4% compared with 64.9% for XNOR under the reported comparison (Table 3).
- The designs transfer to the evaluated point-cloud architectures, including PointNet++, DGCNN, and PointConv (Table 3).
- First/last-layer precision and batch-normalization choices materially affect deployment accuracy, storage, and latency (Appendix B, Table 4).

## Original implementation and usage

Created by [Haotong Qin](https://htqin.github.io/), [Zhongang Cai](https://scholar.google.com/citations?user=WrDKqIAAAAAJ&hl=en), [Mingyuan Zhang](https://scholar.google.com/citations?user=2QLD4fAAAAAJ&hl=en), Yifu Ding, Haiyu Zhao, Shuai Yi, [Xianglong Liu](http://sites.nlsde.buaa.edu.cn/~xlliu/), and [Hao Su](https://cseweb.ucsd.edu/~haosu/) from Beihang University, SenseTime, and UCSD.

![prediction example](https://htqin.github.io/Imgs/ICLR/overview_v1.png)

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

## Citation

Please cite the published paper below. Open paper versions are linked at the top of this README.

```bibtex
@inproceedings{Qin:iclr21,
  title = {{BiPointNet}: Binary Neural Network for Point Clouds},
  author = {Haotong Qin and Zhongang Cai and Mingyuan Zhang and Yifu Ding and Haiyu Zhao and Shuai Yi and Xianglong Liu and Hao Su},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
  url = {https://openreview.net/forum?id=9QLRCVysdlO}
}
```
