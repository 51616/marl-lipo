# Generating Diverse Cooperative Agents by Learning Incompatible Policies

[Paper](https://openreview.net/forum?id=UkU05GOH7_6) | [Project page](https://sites.google.com/view/iclr-lipo-2023)

### Installation

If you don't have conda, install conda first.
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create a new environment and install cuda and torch.
```
conda create -n coop_marl python=3.8
conda activate coop_marl
```

The installer will use CUDA 11.1, so make sure that your current Nvidia driver supports that.
You can install Nvidia driver using `sudo apt install nvidia-driver-515`. You can change the number to install a different version.

If torch can't see your GPU, add this to your `~/.bashrc`.
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
```

Installing dependencies.
```
./install.sh
```

### Training

Examples of training command are in `scripts/`. The commands are based on the best searched hyperparameters of the corresponding algorithm and environment. If you want to run the scripts, make sure that you are currently at the root of the project.

For generalist agents, you can use the following commands:
```
xvfb-run -a python main.py --config_file config/algs/meta/overcooked.yaml \
--env_config_file config/envs/overcooked.yaml --config {"partner_dir": ["..."], "render": 0}
```
where `partner_dir` is the path to the training partners e.g., `training_partners_8/overcooked_full_divider_salad_4/trajedi/20220919-233301`.
Generalist agents can only be trained after you obtained the training partners.

### BibTeX
```
@inproceedings{charakorn2023generating,
title={Generating Diverse Cooperative Agents by Learning Incompatible Policies},
author={Rujikorn Charakorn and Poramate Manoonpong and Nat Dilokthanakul},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UkU05GOH7_6}
}
```
