import torch
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from utils import gif75speaker
import numpy as np
from torchvision import transforms
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str)
# parser.add_argument('--local-rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument("--use_amp", action='store_true', default=False)
# opt = parser.parse_args()
# print(opt)


unet1 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()
unet2 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

# elucidated imagen, which contains the unets above (base unet and super resoluting ones)

imagen = ElucidatedImagen(
    text_embed_dim=1024,
    unets = (unet1, unet2),
    image_sizes = (64, 64),
    random_crop_sizes = (None, 16),
    temporal_downsample_factor = (1, 1),        # in this example, the first unet would receive the video temporally downsampled by 2x
    num_sample_steps = 10,
    cond_drop_prob = 0.1,
    sigma_min = 0.002,                          # min noise level
    sigma_max = (80, 160),                      # max noise level, double the max noise level for upsampler
    sigma_data = 0.5,                           # standard deviation of data distribution
    rho = 7,                                    # controls the sampling schedule
    P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
).cuda()

# mock videos (get a lot of this) and text encodings from large T5

# texts = [
#     'a whale breaching from afar',
#     'young girl blowing out candles on her birthday cake',
#     'fireworks with blue and green sparkles',
#     'dust motes swirling in the morning sunshine on the windowsill'
# ]

# videos = torch.randn(4, 3, 10, 32, 32).cuda() # (batch, channels, time / video frames, height, width)

# feed images into imagen, training each unet in the cascade
# for this example, only training unet 1
FROM_PRETRAIN = False
# imagen.load_state_dict(torch.load('/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/imagen-video-audio960h-IgnoreTime-169'))
imagen.load_state_dict(torch.load('/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/imagen-video-audio60pho-IgnoreTime-10frames-798'))
imagen.train()

trainer = ImagenTrainer(imagen,
    split_valid_from_train = True, # whether to split the validation dataset from the training
    dl_tuple_output_keywords_names = ('images', 'text_embeds', 'cond_video_frames')
).cuda()

(gifs, aud_emb, cond_video_frames) = next(iter(gif75speaker(img_per_gif=10, audio_path = './datasets/audios-eng-pho')))
# you can also ignore time when training on video initially, shown to improve results in video-ddpm paper. eventually will make the 3d unet trainable with either images or video. research shows it is essential (with current data regimes) to train first on text-to-image. probably won't be true in another decade. all big data becomes small data
# for i in range(1,200000):
trainer.add_train_dataset(gif75speaker(audio_path = './datasets/audios-eng-pho'), batch_size = 2)

for i in range(100000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 2, ignore_time = False)
    print(f'loss: {loss}')

    if not (i % 50):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 2)
        print(f'valid loss: {valid_loss}')

    if not (i % 200) and trainer.is_main: # is_main makes sure this can run in distributed
        videos = trainer.sample(text_embeds = aud_emb.unsqueeze(0), video_frames = 10, stop_at_unet_number  = 1, batch_size = 1, cond_video_frames=cond_video_frames.unsqueeze(0))
        imgs = torch.transpose(videos[0], 0, 1)
        imgs = [transforms.ToPILImage()(img) for img in imgs]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        imgs[0].save(f'./gif_samples/gif-sample-audio60pho-notIgnoreTime-10frames-{i // 200}.gif', save_all=True, append_images=imgs[1:], duration=20, loop=0)
        torch.save(imagen.state_dict(), f'./checkpoints/imagen-video-audio60pho-notIgnoreTime-10frames-{i // 200}')