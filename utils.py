# from PIL import Image
from torch.utils.data import Dataset
import glob
from torchvision import transforms
import torch
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from PIL import Image, ImageSequence
import numpy as np
import random
# import glob
from PIL import Image, ImageSequence
# from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(64),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
}

class span75speaker(Dataset):
    def __init__(self, image_path = './datasets/images', audio_path = './datasets/audios', transform=None, target_transform=None):
        self.images = glob.glob(f'{image_path}/*')  # Could be a list: ['./train/input/image_1.bmp', './train/input/image_2.bmp', ...]
        # self.audios = glob.glob(f'{audio_path}/*')  # Could be a nested list: [['./train/GT/image_1_1.bmp', './train/GT/image_1_2.bmp', ...], ['./train/GT/image_2_1.bmp', './train/GT/image_2_2.bmp', ...]]
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        image_name = self.images[index].split('/')[-1].split('.')[0].split('-')
        img = Image.open(self.images[index])
        if self.transform:
            img = self.transform(img)

        aud_embs = torch.load(f'./datasets/audios/{image_name[0]}.pt')
        aud_emb = aud_embs[:,min(int(image_name[-1]), aud_embs.size(1) - 1),:]

        return (img, aud_emb)

    def __len__(self):
        return len(self.images)
    
def create_audio_emds(in_path = '/mnt/c/Users/PCM/Dropbox/span/sub006/2drt/audio', out_path = './datasets/audios'):
    aud_list = glob.glob(f'{in_path}/*')
    for path in aud_list:
        name = path.split('/')[-1].split('.')[0]
        input_audio, sample_rate = librosa.load(f"{in_path}/{name}.wav",  sr=16000)

        model_name = "facebook/wav2vec2-large-xlsr-53"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)

        i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            o = model(i.input_values)
        torch.save(o.last_hidden_state, f'{out_path}/{name}.pt')

def add_noise_video(sample_img, image_sizes = (64, 64), timesteps=1000, times = 100):
    ret = []
    for i in range(10):
        x_noisy, _, _, _ = add_noise(sample_img[:,:,i,:].cpu(), image_sizes = image_sizes, timesteps=timesteps, times = times)
        ret.append(x_noisy)
    return torch.stack(ret, axis=2)

class gif75speaker(Dataset):
    def __init__(self, image_path = './datasets/gifs', audio_path = './datasets/audios', transform=None, target_transform=None, img_per_gif = 10, audio_pooling=False, mode='train', seed=2024):
        self.gifs = glob.glob(f'{image_path}/*')  # Could be a list: ['./train/input/image_1.bmp', './train/input/image_2.bmp', ...]
        if(mode == 'train'):
            pass
        elif(mode == 'test-unseensubject'):
            self.gifs = glob.glob(f'{image_path}/sub0[6-7][1-9]*')
        elif(mode == 'test-unseenaudio'):
            self.gifs = glob.glob(f'{image_path}/*_topic[2-5]_*')
        elif(mode == 'test-unseenboth'):
            self.gifs = glob.glob(f'{image_path}/sub0[6-7][1-9]*_topic[1-5]_*')
        else:
            assert False, f"No mode {mode} Found. Try [train, test-unseensubject, test-unseenaudio, test-unseenboth]"
        random.Random(seed).shuffle(self.gifs)
        self.audios = audio_path #glob.glob(f'{audio_path}/*')  # Could be a nested list: [['./train/GT/image_1_1.bmp', './train/GT/image_1_2.bmp', ...], ['./train/GT/image_2_1.bmp', './train/GT/image_2_2.bmp', ...]]
        self.transform = transform
        self.target_transform = target_transform
        self.img_per_gif = img_per_gif
        self.audio_pooling = audio_pooling

    def __getitem__(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0].split('-')

        with Image.open(self.gifs[index]) as im:
            gif = self.load_frames(im)
        # gif = Image.open(self.images[index])

        aud_embs = torch.load(f'{self.audios}/{gifs_name[0]}.pt')
        aud_emb = aud_embs[:,int(gifs_name[-1]):int(gifs_name[-1]) + self.img_per_gif,:]
        if(self.audio_pooling):
            aud_emb = torch.mean(aud_emb, axis=0).unsqueeze(0)
        gif = torch.transpose(torch.stack([transforms.ToTensor()(i) for i in gif[:self.img_per_gif]]), 0,1)
        # low_res = add_noise_video(gif.unsqueeze(0), image_sizes = (64, 64), timesteps=1000, times = 200)
        return (gif, aud_emb[0], gif[:,0:1,:])

    def __len__(self):
        return len(self.gifs)

    def get_start_index(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0].split('-')
        return int(gifs_name[-1])
        
    def get_names(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0]
        return gifs_name
    
    def get_path(self, index):
        gifs_name = self.gifs[index]#.split('/')[-1].split('.')[0]
        return gifs_name
    
    def get_audio_emb(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0].split('-')
        aud_embs = torch.load(f'{self.audios}/{gifs_name[0]}.pt')
        return aud_embs
    
    def load_frames(self, image: Image, mode='RGB'):
        # ret = 
        # if self.transform:
        #     gif = self.transform(gif)
        return np.array([
            np.array(frame.convert(mode))
            for frame in ImageSequence.Iterator(image)
        ])
    

class gif75speaker_res(Dataset):
    def __init__(self, image_path = './datasets/gifs', audio_path = './datasets/audios', transform=None, target_transform=None, img_per_gif = 10, audio_pooling=False):
        self.gifs = glob.glob(f'{image_path}/*')  # Could be a list: ['./train/input/image_1.bmp', './train/input/image_2.bmp', ...]
        self.audios = audio_path #glob.glob(f'{audio_path}/*')  # Could be a nested list: [['./train/GT/image_1_1.bmp', './train/GT/image_1_2.bmp', ...], ['./train/GT/image_2_1.bmp', './train/GT/image_2_2.bmp', ...]]
        self.transform = transform
        self.target_transform = target_transform
        self.img_per_gif = img_per_gif
        self.audio_pooling = audio_pooling

    def __getitem__(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0].split('-')

        with Image.open(self.gifs[index]) as im:
            gif = self.load_frames(im)
        # gif = Image.open(self.images[index])

        aud_embs = torch.load(f'{self.audios}/{gifs_name[0]}.pt')
        aud_emb = aud_embs[:,int(gifs_name[-1]):int(gifs_name[-1]) + self.img_per_gif,:]
        if(self.audio_pooling):
            aud_emb = torch.mean(aud_emb, axis=0).unsqueeze(0)
        gif = torch.transpose(torch.stack([transforms.ToTensor()(i) for i in gif[:self.img_per_gif]]), 0,1)
        low_res = add_noise_video(gif.unsqueeze(0), image_sizes = (64, 64), timesteps=1000, times = 200)
        return (gif, aud_emb[0], low_res[0])

    def __len__(self):
        return len(self.gifs)

    def get_names(self, index):
        gifs_name = self.gifs[index].split('/')[-1].split('.')[0]
        return gifs_name
    
    def load_frames(self, image: Image, mode='RGB'):
        # ret = 
        # if self.transform:
        #     gif = self.transform(gif)
        return np.array([
            np.array(frame.convert(mode))
            for frame in ImageSequence.Iterator(image)
        ])

import torch
from torch import nn
from imagen_pytorch.imagen_pytorch import cast_uint8_images_to_float, resize_image_to, normalize_neg_one_to_one, GaussianDiffusionContinuousTimes, cast_tuple, pad_tuple_to_length, default
from utils import *
# import matplotlib.pyplot as plt

def add_noise(images, image_sizes = (64, 64), timesteps=1000, times = 500):
    times = torch.tensor([times/timesteps])
    assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
    images = cast_uint8_images_to_float(images)
    assert images.dtype == torch.float or images.dtype == torch.half, f'images tensor needs to be floats but {images.dtype} dtype found instead'
    assert images.shape[1] == 3
    images = resize_image_to(images, image_sizes)
    x_start = images#normalize_neg_one_to_one(images)

    timesteps = cast_tuple(timesteps, 2)

    # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets
    noise_schedules = 'cosine'
    noise_schedules = cast_tuple(noise_schedules)
    noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
    noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'linear')

    noise_scheduler_klass = GaussianDiffusionContinuousTimes
    noise_schedulers = nn.ModuleList([])
    noise = default(None, lambda: torch.randn_like(x_start))

    for timestep, noise_schedule in zip(timesteps, noise_schedules):
        noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
        noise_schedulers.append(noise_scheduler)
    x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)
    return x_noisy, log_snr, alpha, sigma

def get_path_of_pretrained(AUDIO_EMB, POOLING):
    path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wav2vec2/lv60/ImagenVideo-Modelwav2vec2-l60-PoolingFalse-IgnoreTimeFalse-TwoStepTrue-100'
    emb_len = 1024
    if(AUDIO_EMB == 'wav2vec2-l60-pho'):
        if(POOLING):
            path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wav2vec2/pooling/ImagenVideo-Modelwav2vec2-l60-pho-PoolingTrue-IgnoreTimeFalse-TwoStepTrue-100'
        else:
            path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wav2vec2/phoneme/ImagenVideo-Modelwav2vec2-l60-pho-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-100'
    elif(AUDIO_EMB == 'wav2vec2-l60'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wav2vec2/lv60/ImagenVideo-Modelwav2vec2-l60-PoolingFalse-IgnoreTimeFalse-TwoStepTrue-100'
    elif(AUDIO_EMB == 'wav2vec2-base'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wav2vec2/base/ImagenVideo-Modelwav2vec2-base-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-100'
        emb_len = 768
    elif(AUDIO_EMB == 'hubert-base'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/hubert/base/ImagenVideo-Modelhubert-base-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-96'
        emb_len = 768
    elif(AUDIO_EMB == 'hubert-large'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/hubert/large/ImagenVideo-Modelhubert-large-PoolingFalse-IgnoreTimeFalse-TwoStepTrue-100'
    elif(AUDIO_EMB == 'wavlm-base'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wavlm/base/ImagenVideo-Modelwavlm-base-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-100'
        emb_len = 768
    elif(AUDIO_EMB == 'wavlm-large'):
        path = '/mnt/c/Users/PCM/Documents/GitHub/SPAN-rtmri/checkpoints/wavlm/large/ImagenVideo-Modelwavlm-large-PoolingFalse-IgnoreTimeFalse-TwoStepFalse-100'
    else:
        assert False, "No AUDIO_EMB name found! Try different name"
    return path, emb_len


def load_frames(image: Image, mode='RGB'):
    # ret = 
    # if self.transform:
    #     gif = self.transform(gif)
    return np.array([
        np.array(frame.convert(mode))
        for frame in ImageSequence.Iterator(image)
    ])

def load_frames_tensor(im: Image, mode='RGB', video_len=10):
    return torch.stack([transforms.ToTensor()(np.array(frame.convert('RGB'))) for frame in ImageSequence.Iterator(im)])[:video_len]

def get_videos_from_folder(paths):
    synthetic_batch = []
    for names in paths:
        with Image.open(names) as im:
            gif = load_frames_tensor(im)
            # gif = load_frames_tensor(im)
            synthetic_batch.append(gif)
    synthetic_batch = torch.stack(synthetic_batch)
    return synthetic_batch