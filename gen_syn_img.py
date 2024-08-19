import torch
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from utils import gif75speaker, get_path_of_pretrained
import numpy as np
from torchvision import transforms
import subprocess

AUDIO_EMB = 'hubert-base'
POOLING = False
MODE = 'test-unseensubject' #Select {test-unseenaudio, test-unseensubject, test-unseenboth}
LEN_GEN_IMGS = 500 # Number of generated images for evaluation
PATH_2_PRETRAINED, LEN_AUDIO_EMB = get_path_of_pretrained(AUDIO_EMB, POOLING)
print(PATH_2_PRETRAINED)

dataset_75speaker = gif75speaker(image_path = './datasets/preprocessed_dataset/test', 
                                img_per_gif = 10, 
                                audio_path = f'./datasets/preprocessed_dataset/{AUDIO_EMB}', 
                                audio_pooling = POOLING,
                                mode = MODE)

unet1 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()
unet2 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

imagen = ElucidatedImagen(
    text_embed_dim = LEN_AUDIO_EMB,
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

imagen.load_state_dict(torch.load(PATH_2_PRETRAINED))
trainer = ImagenTrainer(imagen,
    split_valid_from_train = True, # whether to split the validation dataset from the training
    dl_tuple_output_keywords_names = ('images', 'text_embeds', 'cond_video_frames')
).cuda()

# !rm -rf ./generated_images
# !mkdir ./generated_images
# subprocess.call(f"rm -rf ./generated_images/{AUDIO_EMB}", shell=True)
# subprocess.call(f"mkdir ./generated_images/{AUDIO_EMB}", shell=True)
# subprocess.call(f"rm -rf ./generated_images/{AUDIO_EMB}/{MODE}", shell=True)
# subprocess.call(f"mkdir ./generated_images/{AUDIO_EMB}/{MODE}", shell=True)

for i in range(LEN_GEN_IMGS):
    (_, aud_emb, cond_video_frames) = dataset_75speaker[i]
    # real_path.append(dataset_75speaker.get_path(i))
    print(f'{dataset_75speaker.get_names(i)}')
    videos = trainer.sample(text_embeds = aud_emb.unsqueeze(0), video_frames = 10, stop_at_unet_number  = 1, batch_size = 1, cond_video_frames=cond_video_frames.unsqueeze(0))
    imgs = torch.transpose(videos[0], 0, 1)
    imgs = [transforms.ToPILImage()(img) for img in imgs]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    # model_name = opt.audio_path.split('/')[-1]
    imgs[0].save(f'./generated_images/{AUDIO_EMB}/{MODE}/{dataset_75speaker.get_names(i)}.gif', save_all=True, append_images=imgs[1:], duration=10, loop=0)