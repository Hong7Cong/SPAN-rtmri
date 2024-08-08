# SPAN-rtmri

## Setup enviroment

## Prepare dataset
Create gifs for training and test from original SPAN dataset
```
python create_gifs_dataset.py
                --path2span /mnt/c/Users/PCM/Dropbox/span
                --target_dir ./datasets/preprocessed_dataset/
```
Create audio embeddings from a pretrained model
```
python create_audio_embeddings.py
                --path2span /mnt/c/Users/PCM/Dropbox/span
                --target_dir ./datasets/preprocessed_dataset/audio_embs
                --model_name Wav2Vec2
                --model_size large
                --pretrain_on None
```