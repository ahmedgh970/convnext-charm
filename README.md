# ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression
Official implementation of ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression.
Paper link on [ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression](Coming Soon)

* [ConvNeXt-ChARM](#convnext-charm)
  * [Disclaimer & Documentation](#disclaimer-documentation)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [CLI-Usage](#cli-usage)
  * [License](#license)
    
<!-- /code_chunk_output -->

## Tags
<code>Swin Transformer</code>, <code>ConvNeXt</code>, <code>Learning-based Codecs</code>, <code>Image Compression</code>, <code>TensorFlow</code>

## Disclaimer & Documentation
Please do not hesitate to open an issue to inform of any problem you may find within this repository. Also, you can [email me](ahmed.ghorbel888@gmail.com) for questions or comments. 
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
├── testsets/
│   └── CLIC22/ - CLIC22 dataset
│
├── utilities/
│   └── utils.py/ - utility scripts
│   └── convNext.py/ - ConvNeXt block layers
│   └── swinTransformer.py/ - Swin Transformer block layers
|
├── ckpts/  - Checkpoints folder
|
├── results/  - Evaluation results folder
│
└── figures/ - Documentation figures
```

## CLI Usage
Every model can be trained and tested individually using:
* <code>Python convnext-charm.py train</code>
* <code>Python convnext-charm.py evaluate</code>

## Citation
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
