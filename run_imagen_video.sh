# cd /mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri
# conda activate genai
python imagen-video-training.py \
--audio_path ./datasets/preprocessed_dataset/wav2vec2-l60 \
--gif_path ./datasets/preprocessed_dataset/train \
--pretrained_path /mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/ImagenVideo-Modelwav2vec2-l60-pho-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-1 \
# --from_pretrained