# The source code for ICML2021 submission #7882 "When Does Data Augmentation Help With Membership Inference Attacks?"


This is research code and it is not easy to run it, right now most of the selections (e.g., which dataset) is hardcoded. My biggest motivation for sharing it is for reviewers to see how certain methods I mention in the submission are implemented, such as LRC or the membership inference attacks.


**Requirements:**
pytorch opacus 0.9 (https://github.com/pytorch/opacus/releases/tag/v0.9.0)
pytorch 1.7.1
torchvision 0.8.2
tqdm
CUDA 11.0
CUDNN 8.0
sns

**Source code files:**
aux_funcs.py --- contains auxiliary functions for optimizers, regularization methods, datasets, loaders etc.
augmentation_combinations.py --- contains the implementation of the LRC metric in Section 6.
mi_attacks.py --- contains the implementations of the two MIAs and methods to find and attack models in a hardcoded path
collect_results.py --- contains the methods to collect the results after models are trained and the MIAs are applied
smooth_distillation.py --- contains the wrapper to run the experiment in Section 5 last part about distillation and label smoothing
train_models.py --- contains the wrappers to train all the models with augmentation using hardcoded hyper-parameters
models.py --- contains the definitions of the architectures we experiment with, the methods to train models with DP or with augmentation

### Check out experiments.ipynb to see the code that generates the figures and the data for the tables in the paper, the notebook includes these figures already.


