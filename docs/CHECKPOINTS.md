
# Prepare checkpoints

This document contains commands for preparing model checkpoints and the final checkpoint organization structure.

### Download checkpoints
Please run the following commands in your terminal to download the checkpoints for each model.
```
mkdir checkpoints

# ---------------------------- T2V ----------------------------

# ---- CogVideo ----
mkdir -p checkpoints/cogvideo
cd checkpoints/cogvideo
git clone https://huggingface.co/THUDM/CogVideoX-2b
git clone https://huggingface.co/THUDM/CogVideoX-5b
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
mkdir  CogVideoX1.5-5B-SAT
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='THUDM/CogVideoX1.5-5B-SAT', ignore_patterns=['t5*'], local_dir='CogVideoX1.5-5B-SAT')"
# VAE for CogVideoX-2B, 5B, 5B-I2V
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vaesat.zip
unzip vaesat.zip -d vaesat
rm -rf vaesat.zip
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
rm -rf transformer.zip
cd ..
mkdir CogVideoX-5b-sat
cd CogVideoX-5b-sat
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1"
mv 'index.html?p=%2F1%2FCogVideoX-5B-transformer.tar.gz&dl=1' transformer.tar.gz
tar -xzvf transformer.tar.gz
mv 'CogVideoX-5B-transformer' transformer
rm transformer.tar.gz
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2Flatest&dl=1"
mv 'index.html?p=%2Flatest&dl=1' latest
cd ..
mkdir CogVideoX-5B-I2V
cd CogVideoX-5B-I2V
wget "https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/files/?p=%2F1%2Fmp_rank_00_model_states.zip&dl=1"
mkdir -p transformer/1
mv "index.html?p=%2F1%2Fmp_rank_00_model_states.zip&dl=1" "transformer/1/transformer.zip"
unzip "transformer/1/transformer.zip" -d "transformer/1/"
rm -rf transformer/1/transformer.zip
wget "https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/files/?p=%2Flatest&dl=1"
mv 'index.html?p=%2Flatest&dl=1' "transformer/latest"
cd ..

# ---- Open-Sora ----
mkdir -p checkpoints/open-sora/t2v_v10
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
wget https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth -P checkpoints/open-sora/t2v_v10/
#
mkdir -p checkpoints/open-sora/t2v_v11
cd checkpoints/open-sora/t2v_v11
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2
git clone https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3
cd ../../..
#
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3
mkdir -p checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-VAE-v1.2
wget https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/resolve/main/model.safetensors -P checkpoints/open-sora/t2v_v12/OpenSora-STDiT-v3


# ---- Videocrafter ----
mkdir checkpoints/videocrafter/

mkdir checkpoints/videocrafter/t2v_v2_512
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v2_512  # videocrafter2-t2v-512

mkdir checkpoints/videocrafter/t2v_v1_1024
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt -P checkpoints/videocrafter/t2v_v1_1024 # videocrafter1-t2v-1024


# ---------------------------- I2V ----------------------------
# ---- Dynamicrafter ----
mkdir checkpoints/dynamicrafter/
mkdir checkpoints/dynamicrafter/i2v_576x1024

wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt -P checkpoints/dynamicrafter/i2v_576x1024  # dynamicrafter-i2v-1024

# ---- Videocrafter ----
mkdir checkpoints/videocrafter/
mkdir checkpoints/videocrafter/i2v_v1_512

wget https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt -P checkpoints/videocrafter/i2v_v1_512 # videocrafter1-i2v-512

# ---- Stable Diffusion checkpoint for VC2 Training ----
mkdir checkpoints/stablediffusion/
mkdir checkpoints/stablediffusion/v2-1_512-ema

wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt -P checkpoints/stablediffusion/v2-1_512-ema

```

### Checkpoint Orgnization Structure
After downloading, the model checkpoints should be placed as follows:  

```
VideoTuna/
    └── checkpoints/
        ├── cogvideo/
        │   └── CogVideoX-2b/   
        │   └── CogVideoX-5b/        
        │   └── CogVideoX-5b-I2V/        
        ├── dynamicrafter/
        │   └── i2v_576x1024/
        │       └── model.ckpt
        ├── videocrafter/
        │   ├── t2v_v2_512/
        │   │   └── model.ckpt
        │   ├── t2v_v1_1024/
        │   │   └── model.ckpt
        │   └── i2v_v1_512/
        │       └── model.ckpt
        └── open-sora/
            ├── t2v_v10/
            │   ├── OpenSora-v1-16x256x256.pth
            │   └── OpenSora-v1-HQ-16x512x512.pth
            ├── t2v_v11/
            │   ├── OpenSora-STDiT-v2-stage2/
            │   └── OpenSora-STDiT-v2-stage3/
            └── t2v_v12/
                ├── OpenSora-STDiT-v3/
                └── OpenSora-VAE-v1.2/
```

If you do not follow these locations, please modify the default checkpoint path argument during training/inference.
