# Installation

This project has been tested with Ubuntu 18.04 using the following main libraries: `pytorch=1.4` and `MinkowskiEngine=0.4.1`. 
You will need a full installation of CUDA 10.1 in order to compile MinkowskiEngine v0.4.1.

One possible way to install the required libraries through Anaconda and Pip is by using the following commands:

```
# create a conda environment and install requirements for MinkowskiEngine v0.4.1

conda create -n points_separation python=3.7
conda activate points_separation
conda install numpy openblas
conda install pytorch torchvision -c pytorch

# Install MinkowskiEngine v0.4.1

wget https://github.com/NVIDIA/MinkowskiEngine/archive/refs/tags/v0.4.1.tar.gz
tar -zxvf v0.4.1.tar.gz
cd MinkowskiEngine-0.4.1/
python setup.py install

# install additional libraries

conda install scipy
conda install -c conda-forge imageio
conda install -c conda-forge mir_eval
pip install open3d
conda install -c pytorch torchaudio
conda install -c conda-forge librosa
conda install -c anaconda scikit-image
conda install -c conda-forge opencv

```
With this newly created environment, you can start using the repo.
