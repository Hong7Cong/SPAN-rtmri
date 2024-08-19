
# python imagen-video-training.py \
# --audio_path ./datasets/preprocessed_dataset/wav2vec2-base \
# --gif_path ./datasets/preprocessed_dataset/train \
# --audio_embed_dim 768
# --pretrained_path /mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/ImagenVideo-Modelwav2vec2-l60-pho-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-1 \
# --from_pretrained

# python create_audio_embeddings.py --path2span /mnt/c/Users/PCM/Dropbox/span --target_dir ./datasets/preprocessed_dataset/wav2vec2-base --model_name Wav2Vec2 --model_size base --pretrain_on None

python imagen-video-training.py \
--audio_embed_dim 1024 \
--audio_path ./datasets/preprocessed_dataset/hubert-large \
--gif_path ./datasets/preprocessed_dataset/train \
--audio_pooling 
# --from_pretrained \
# --pretrained_path checkpoints/ImagenVideo-Modelwav2vec2-l60-pho-PoolingTrue-IgnoreTimeFalse-TwoStepTrue-10