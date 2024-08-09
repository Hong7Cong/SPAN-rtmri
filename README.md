# SPAN-rtmri

## Setup enviroment

## Prepare dataset
Create gifs from original SPAN dataset for training and test
```
python create_gifs_dataset.py
                --path2span /mnt/c/Users/PCM/Dropbox/span
                --target_dir ./datasets/preprocessed_dataset/
```
Create audio embeddings from a pretrained model. Each pretrained model should have seperated target_dir folder.
```
python create_audio_embeddings.py
                --path2span /mnt/c/Users/PCM/Dropbox/span
                --target_dir ./datasets/preprocessed_dataset/audio_embs
                --model_name Wav2Vec2
                --model_size large
                --pretrain_on None
```
Train diffusion model
```
python imagen-video-training.py 
--audio_path ./datasets/preprocessed_dataset/audio_embs 
--gif_path ./datasets/preprocessed_dataset/train
```