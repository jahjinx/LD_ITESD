# !/usr/bin/env zsh
SECONDS=0

source ~/opt/anaconda3/etc/profile.d/conda.sh

# Set miniconda environment path
ENV_PATH="$HOME/opt/anaconda3/envs"

echo "The folowing environments are installed:$(ls $ENV_PATH)"

read -rp "Enter the environment name: " env_name

dir_path="$ENV_PATH/$env_name"

if [ -d "$dir_path" ];
then
    echo "$env_name environment exists."
	sleep 1
	exit 0
else
	echo "$env_name environment does not exist."
	sleep 1 
	echo "Will create the $env_name conda environment"
fi

CONDA_SUBDIR=osx-arm64 conda create -n $env_name python=3.9 -c conda-forge

env_location=$(conda env list | grep $env_name)

echo ">>> The environemt $env_name has been created"
echo ">>> The environemt is at $env_location"



printf "Activating the environment...\\n"
conda activate $env_name

echo "Configuring ARM Variables"
conda env config vars set CONDA_SUBDIR=osx-arm64
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

printf "Deactivating the environment...\\n"
conda deactivate

printf "Re-Activating the environment...\\n"
conda activate $env_name

echo "Installing Packages"
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpupip3 install transformers
pip3 install torchinfo
pip3 install scikit-learn
pip3 install pandas
pip3 install tqdm
conda install -n $env_name ipykernel --update-deps --force-reinstall

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."