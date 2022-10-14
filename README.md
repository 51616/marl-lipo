# Coop MARL

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
