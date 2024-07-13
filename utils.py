from PIL import Image
from torch.utils.data import Dataset
import glob
from torchvision import transforms
import torch
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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