# SPAN-rtmri Audio-Guided Diffusion Model

## Setup enviroment
```
conda env create --file=environments.yml
conda activate genai
```

## Preparing dataset
Create preprocessed dataset folder
```
mkdir ./datasets/preprocessed_dataset/
```
Create gifs from original SPAN dataset for training and test
```
python create_gifs_dataset.py
                --path2span /mnt/c/Users/PCM/Dropbox/span
                --target_dir ./datasets/preprocessed_dataset/
```
Create audio embeddings from a pretrained model. Each pretrained model should have seperated target_dir folder.
```
python create_audio_embeddings.py --path2span /mnt/c/Users/PCM/Dropbox/span --target_dir ./datasets/preprocessed_dataset/hubert-large --model_name Hubert --model_size large --pretrain_on None
```

## Training
Train diffusion model, save models to ./checkpoints/ and save samples per epoch to ./gif_samples
```
python imagen-video-training.py 
                --audio_path ./datasets/preprocessed_dataset/audio_embs 
                --audio_embed_dim 1024
                --from_pretrained False
                --ignore_time False
                --audio_pooling False
                --gif_path ./datasets/preprocessed_dataset/train
```