# Conditional Causal Representation Learning for Heterogeneous Single-cell RNA Data Integration and Prediction
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains code, data, tables and plots to support data analyses and reproduce results from the paper ComGRN: multi-domain single-cell data analysis guided by Common Gene Regulatory Network.
- [Abstract](#abstract)
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [Data Availability](#data-availability)
- [License](#license)


# Abstract
Single-cell sequencing technology provides deep insights into gene activity at the individual cell level, facilitating the study of gene regulatory mechanisms. However, observed gene expression is often influenced by confounding factors such as batch effects, perturbations, and spatial position, which obscure the true gene regulatory network that governs the cell’s intrinsic state. To address these challenges, we propose scConCRL, a novel conditional causal representation learning framework designed to extract true gene regulatory relationships independent of confounding information. By considering both fine-grained molecular gene variables and coarse-grained latent domain variables, scConCRL not only uncovers the intrinsic biological signals but also models the complex relationships between these variables. This dual function enables the separation of genuine cellular states from domain information, providing valuable insights for downstream analyses and biological discovery. We demonstrate the effectiveness of our model on multi-domain datasets from different platforms and perturbation conditions, showing its ability to accurately disentangle confounding influences and discover novel gene relationships. Extensive comparisons across various scenarios illustrate the superior performance of scConCRL in several tasks compared to existing methods. 

# Overview
<div align=center>
<img src="" height="400" width="800">
</div>

# main file

<pre> 
scConCRL/
├── config/                              # Configuration files
│   ├── demo.yaml                        # Basic demo configuration
│   ├── demo_OOD.yaml                    # OOD experiment config
│   ├── demo_denoise.yaml                # Real data denoising experiment config
│   ├── demo_grn.yaml                    # GRN inference config
│
├── exps/                                # Experiment scripts
│   ├── batch_removal/                   # Batch effect removal experiments
│   ├── GRN_inference/                   # Gene regulatory network inference experiments
│   ├── perturbation_response/           # Perturbation response prediction experiments
│
├── figures/                             # Figures and plots generated from experiments
│
├── scConCRL/                            # Core source code
│   ├── __init__.py
│   ├── data_processing.py               # Data loading and preprocessing
│   ├── eval.py                          # Model evaluation methods
│   ├── model.py                         # Main model architecture
│   ├── network.py                       # Neural network components
│   ├── utils.py                         # Utility functions
│
├── eval.ipynb                           # Jupyter Notebook for evaluation
├── README.md                            # Project description and instructions
</pre>

# System Requirements
## Hardware requirements
`scConCRL` requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 22.04

### Python Dependencies
`scConCRL` mainly depends on the Python scientific stack.
```
numpy
pandas
torch
scanpy
anndata
scikit-learn
scipy
igraph
umap
scib
scib_metrics
leidenalg
tqdm
omegaconf
```
For specific setting, please see requirement.txt.

# Installation Guide
``` 
$ git clone https://github.com/fdu-wangfeilab/scConCRL.git
$ conda create -n scConCRL_env python=3.8.18
$ conda activate scConCRL_env
$ conda env create -f environment.yml
```

# Detailed tutorials with example datasets
The example can be seen in the exps directory.


# Citation

If you find our work useful, please cite our paper:

```
coming soon

```



