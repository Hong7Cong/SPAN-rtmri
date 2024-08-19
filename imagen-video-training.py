import torch
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from utils import gif75speaker
import numpy as np
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_sizes", type=int, default=64)
parser.add_argument("--audio_embed_dim", type=int, default=1024)
parser.add_argument("--from_pretrained", action='store_true')
parser.add_argument("--pretrained_path", type=str, default='./checkpoints/ImagenVideo-Modelaudios-eng-pho-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-65')
parser.add_argument("--ignore_time", action='store_true')
parser.add_argument("--audio_pooling", action='store_true')
parser.add_argument("--audio_path", type=str, default='./datasets/preprocessed_dataset/wav2vec2-l60-pho', help="path to pre-extracted")
parser.add_argument("--gif_path", type=str, default='./datasets/gifs', help="path to processed gifs for training")
parser.add_argument("--epoch", type=int, default=22000, help="Number of epoch for training (default 50000)")
parser.add_argument("--save_per", type=int, default=200, help="save per epoch")
parser.add_argument("--use_amp", action='store_true', default=False)
opt = parser.parse_args()
print(opt)


unet1 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()
unet2 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

# elucidated imagen, which contains the unets above (base unet and super resoluting ones)
imagen = ElucidatedImagen(
    text_embed_dim=opt.audio_embed_dim,
    unets = (unet1, unet2),
    image_sizes = (opt.image_sizes, opt.image_sizes),
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

# feed images into imagen, training each unet in the cascade
# for this work, only training unet 1... because unet 2 is for super-resolution
START_EPOCH = 0
FROM_PRETRAIN = opt.from_pretrained
if(FROM_PRETRAIN):
    print(f'---LOAD from pretrained {opt.pretrained_path}---')
    # imagen.load_state_dict(torch.load('/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/imagen-video-audio960h-IgnoreTime-169'))
    # imagen.load_state_dict(torch.load('/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/imagen-video-audio60phoAVG-IgnoreTime-10frames-120'))
    imagen.load_state_dict(torch.load(opt.pretrained_path))
    START_EPOCH = int(opt.pretrained_path.split('-')[-1])*opt.save_per
    imagen.train()

trainer = ImagenTrainer(imagen,
    split_valid_from_train = True, # whether to split the validation dataset from the training
    dl_tuple_output_keywords_names = ('images', 'text_embeds', 'cond_video_frames')
).cuda()

dataset_75speaker = gif75speaker(image_path = opt.gif_path, img_per_gif=10, audio_path = opt.audio_path, audio_pooling=opt.audio_pooling)
(gifs, aud_emb, cond_video_frames) = next(iter(dataset_75speaker))
# you can also ignore time when training on video initially, shown to improve results in video-ddpm paper. eventually will make the 3d unet trainable with either images or video. research shows it is essential (with current data regimes) to train first on text-to-image. probably won't be true in another decade. all big data becomes small data

trainer.add_train_dataset(dataset_75speaker, batch_size = 2)

for i in range(START_EPOCH, opt.epoch):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 2, ignore_time = opt.ignore_time)
    print(f'loss: {loss}')

    if not (i % 100):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 2)
        print(f'valid loss at epoch {i}: {valid_loss}')

    if not (i % opt.save_per) and trainer.is_main: # is_main makes sure this can run in distributed
        videos = trainer.sample(text_embeds = aud_emb.unsqueeze(0), video_frames = 10, stop_at_unet_number  = 1, batch_size = 1, cond_video_frames=cond_video_frames.unsqueeze(0))
        imgs = torch.transpose(videos[0], 0, 1)
        imgs = [transforms.ToPILImage()(img) for img in imgs]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        model_name = opt.audio_path.split('/')[-1]
        imgs[0].save(f'./gif_samples/SampleGeneratedGif-Model{model_name}-Pooling{opt.audio_pooling}-IgnoreTime{opt.ignore_time}-TwoStep{opt.from_pretrained}-{i // opt.save_per}.gif', save_all=True, append_images=imgs[1:], duration=20, loop=0)
        torch.save(imagen.state_dict(), f'./checkpoints/ImagenVideo-Model{model_name}-Pooling{opt.audio_pooling}-IgnoreTime{opt.ignore_time}-TwoStep{opt.from_pretrained}-{i // opt.save_per}')
        # if(FROM_PRETRAIN):
        #     imgs[0].save(f'./gif_samples/SampleGeneratedGif-Model{model_name}-Pooling{opt.audio_pooling}-IgnoreTime{opt.ignore_time}-TwoStep{opt.from_pretrained}-{i // opt.save_per}.gif', save_all=True, append_images=imgs[1:], duration=20, loop=0)
        #     torch.save(imagen.state_dict(), f'./checkpoints/ImagenVideo-Model{model_name}-Pooling{opt.audio_pooling}-IgnoreTime{opt.ignore_time}-TwoStep{opt.from_pretrained}-{i // opt.save_per}')
