# ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression
Official TensorFlow implementation of [ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression](https://arxiv.org/).

* [ConvNeXt-ChARM](#convnext-charm)
  * [Tags](#tags)
  * [Overall ConvNeXt-ChARM Framework](#overall-convnext-charm-framework)
  * [Disclaimer](#disclaimer)
  * [Documentation](#documentation)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [CLI-Usage](#cli-usage)
  * [Rate-Distortion Coding Performance](#rate-distortion-coding-performance)
  * [Citation](#citation)
  * [License](#license)
    
<!-- /code_chunk_output -->

## Tags
<code>Swin Transformer</code> <code>ConvNeXt</code> <code>Learning-based Codecs</code> <code>Image Compression</code> <code>TensorFlow</code>

## Overall ConvNeXt-ChARM Framework
![ConvNeXt-ChARM framework](https://github.com/ahmedgh970/ConvNeXt-ChARM/blob/main/figures/ConvNeXt-ChARM.png)

## Disclaimer
Please do not hesitate to open an issue to inform of any problem you may find within this repository. Also, you can [email me](mailto:ahmed.ghorbel888@gmail.com?subject=[GitHub]) for questions or comments. 

## Documentation
* This repository is built upon the official TensorFlow implementation of [Channel-Wise Autoregressive Entropy Models for Learned Image Compression](https://ieeexplore.ieee.org/abstract/document/9190935). This baseline is referred to as [Conv-ChARM](https://github.com/ahmedgh970/ConvNeXt-ChARM/blob/main/conv-charm.py)
* We provide lightweight versions of the models by removing the latent residual prediction (LRP) transform and slicing latent means and scales, as done in the [Tensorflow reimplementation of SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM) from the original paper [TRANSFORMER-BASED TRANSFORM CODING](https://openreview.net/pdf?id=IDwN6xjHnK8).
* Refer to the [TensorFlow Compression (TFC) library](https://github.com/tensorflow/compression) to build your own ML models with end-to-end optimized data compression built in.
* Refer to the [API documentation](https://www.tensorflow.org/api_docs/python/tfc) for a complete classes and functions description of the TensorFlow Compression (TFC) library.
 

## Requirements
<code>Python >= 3.6</code> <code>tensorflow_compression</code> <code>tensorflow_datasets</code> <code>tensorflow_addons</code> <code>einops</code>

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/ConvNeXt-ChARM/blob/main/requirements.txt).
To install those, run:
```
pip install -r requirements.txt
```

## Folder Structure
``` 
ConvNeXt-ChARM
│
├── conv-charm.py                 # Conv-ChARM Model
├── conv-charm_lightweight.py     # Lightweight Conv-ChARM Model
├── convnext-charm.py             # ConvNeXt-ChARM Model
├── convnext-charm_lightweight.py # Lightweight ConvNeXt-ChARM Model
├── swint-charm.py                # SwinT-ChARM Model
├── swint-charm_lightweight.py    # Lightweight SwinT-ChARM Model
├── utils.py                      # Utility scripts
|
├── testsets/
│   └── CLIC22/                   # CLIC22 dataset
│
├── layers/
│   └── convNext.py/              # ConvNeXt block layers
│   └── swinTransformer.py/       # Swin Transformer block layers
|
├── results/                      # Evaluation results folder
│   └── CLIC22/ 
│       └── ... 
|
└── figures/                      # Documentation figures
```

## CLI Usage
Every model can be trained and tested individually using:
```
python convnext-charm.py train
```
```
python convnext-charm.py evaluate
```

## Rate-Distortion coding performance
![Rate-Distortion coding performance on KODAK](https://github.com/ahmedgh970/ConvNeXt-ChARM/blob/main/figures/rd_performance.png)

Table 1. BD-rate↓ performance of BPG (4:4:4), SwinT-ChARM, and ConvNeXt-ChARM compared to the VTM-18.0 for the four considered datasets.

| Dataset | BPG444 | SwinT-ChARM | ConvNeXt-ChARM |
| --- | --- | --- | --- |
| Kodak   | 20.73% | -3.47%  | -4.90% |
| Tecnick | 27.03% | -6.52%  | -7.56% |
| JPEG-AI | 28.14% | -0.23%  | -1.17% |
| CLIC21  | 26.54% | -5.86%  | -7.36% |
| Average | 25.61% | -4.02%  | -5.24% |


## Citation
If you use this library for research purposes, please cite:
```
@inproceedings{ghorbel2023convnextcharm,
  title={ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression},
  author={Ghorbel, Ahmed and Hamidouche, Wassim and Luce, Morin},
  booktitle={},
  year={2023}
}
```

## License
This project is licensed under the MIT License. See LICENSE for more details
