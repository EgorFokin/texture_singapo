if command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS is installed: $(git-lfs version)"
else
    echo "Git LFS is not installed. Please install it to proceed."
    exit 1
fi


conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub

conda install -y pytorch3d -c pytorch3d

conda install -y xformers -c xformers

pip install -r requirements.txt

mkdir tmp

wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth?download=true -O tmp/control_sd15_canny.pth
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth?download=true -O tmp/control_v11p_sd15_canny.pth

mv tmp/control_sd15_canny.pth easi-tex/models/ControlNet/models
mv tmp/control_v11p_sd15_canny.pth easi-tex/models/ControlNet/models

git clone https://huggingface.co/h94/IP-Adapter tmp/ip_adapter
mv tmp/ip_adapter/models/image_encoder easi-tex/ip_adapter
mv tmp/ip_adapter/models/ip-adapter-plus_sd15.bin easi-tex/ip_adapter

wget https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip -O tmp/pm.zip
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip tmp/pm.zip -d data

rm -rf tmp