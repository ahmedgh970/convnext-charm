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
<code>Swin Transformer</code>, <code>ConvNeXt</code>, <code>Learning-based Codecs</code>, <code>Image Compression</code>, <code>TensorFlow</code>

## Overall ConvNeXt-ChARM Framework
![ConvNeXt-ChARM framework](https://github.com/ahmedgh970/ConvNeXt-ChARM/figures/ConvNeXt-ChARM.pdf)

## Disclaimer
Please do not hesitate to open an issue to inform of any problem you may find within this repository. Also, you can [email me](mailto:ahmed.ghorbel888@gmail.com?subject=[GitHub]) for questions or comments. 

## Documentation
Refer to the [TensorFlow Compression (TFC) library](https://github.com/tensorflow/compression) to build your own ML models with end-to-end optimized data compression built in.
Refer to the [API documentation](https://www.tensorflow.org/api_docs/python/tfc) for a complete description of the classes and functions this package implements.

## Requirements
* <code>Python >= 3.6</code>

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/ConvNeXt-ChARM/requirements.txt).
To install those, run:
```
pip install -r requirements.txt
```

## Folder Structure
``` 
ConvNeXt-ChARM
│
├── conv-charm.py                 # Conv-ChARM Model
├── convnext-charm.py             # ConvNeXt-ChARM Model
├── swint-charm.py                # SwinT-ChARM Model
|
├── testsets/
│   └── CLIC22/                   # CLIC22 dataset
│
├── utilities/
│   └── utils.py/                 # Utility scripts
│   └── convNext.py/              # ConvNeXt block layers
│   └── swinTransformer.py/       # Swin Transformer block layers
|
├── ckpts/                        # Checkpoints folder
|
├── results/                      # Evaluation results folder
│
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
![Rate-Distortion coding performance on KODAK](https://github.com/ahmedgh970/ConvNeXt-ChARM/figures/rd_performance.pdf)

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
