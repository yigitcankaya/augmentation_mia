# The source code for ICML2021 paper [When Does Data Augmentation Help With Membership Inference Attacks?](http://proceedings.mlr.press/v139/kaya21a/kaya21a.pdf)

### [Yigitcan Kaya](http://www.cs.umd.edu/~yigitcan/), [Tudor Dumitras](http://users.umiacs.umd.edu/~tdumitra/) -- University of Maryland, College Park

**Please contact cankaya at umd dot edu for bugs, questions and recommendations.**

**Requirements:**
 * [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-update1-download-archive)
* [CUDNN 8.0](https://developer.nvidia.com/cudnn)
* PyTorch and TorchVision (pip install torch\==1.7.1+cu110 torchvision\==0.8.2+cu110 torchaudio\==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html)
 * Opacus 0.9 - Differential privacy package for PyTorch  (pip install opacus\==0.9.0)
* tqdm (pip install tqdm)

**Source code files and their contents:**
* train_models.py --- contains the wrappers to train all the models with and without augmentation, takes the path to a config file as the command line argument.
* mi_attacks.py --- contains the implementations of the three MIAs in the paper (average, powerful and augmentation-aware), take the path to a config file as the command line argument.
* aux_funcs.py --- contains auxiliary functions for optimizers, regularization methods, datasets, loaders etc.
* loss_rank_correlation.py --- contains the implementation of the LRC metric in Section 6.
* models.py --- contains the definitions of the architectures we experiment with, the methods to train models with DP or with augmentation.
* collect_results.py --- contains the methods to collect the results after models are trained and the MIAs are applied.
* Playground.ipynb --- contains the demonstration of how to use the codebase, train models, apply attacks, collect results etc.
* config.json --- the config file for training and attacking the models, specifies the hyper-parameters, paths, and experiment settings.
