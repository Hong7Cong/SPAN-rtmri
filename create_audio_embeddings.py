import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import argparse
import glob
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--path2span", type=str, default='/mnt/c/Users/PCM/Dropbox/span', help='where the original span dataset located')
parser.add_argument("--target_dir", type=str, default='./datasets/preprocessed_dataset/audio_embs', help='destination to save preprocessed dataset')
parser.add_argument("--model_name", type=str, default='Wav2Vec2', help='model_name')
parser.add_argument("--model_size", type=str, default='large', help='size of audio model')
parser.add_argument("--pretrain_on", type=str, default='phoneme', help='if model pretrained on phoneme')
opt = parser.parse_args()
print(opt)

if(opt.model_name not in ["Wav2Vec2", "Hubert", "WavLM"]):
    assert False, f"No model {opt.model_name} found, try Wave2vec2, Hubert or WavLM instead"
if(opt.model_size not in ["large", "base"]):
    assert False, f"No model size {opt.model_name} found, try large or base instead"
if(opt.pretrain_on not in ["phoneme", "no", "None"]):
    assert False, f"No pretrain_on {opt.model_name} found, try phoneme or None instead"

if(opt.model_name == "Wav2Vec2" and opt.pretrain_on != "phoneme" and opt.model_size == 'large'):
    model_name = "facebook/wav2vec2-large-lv60"
elif(opt.model_name == "Wav2Vec2" and opt.pretrain_on == "phoneme" and opt.model_size == 'large'):
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
elif(opt.model_name == "Wav2Vec2" and opt.model_size == 'base'):
    model_name = "facebook/wav2vec2-base"
elif(opt.model_name == "Hubert" and opt.model_size == 'large'):
    model_name = "facebook/hubert-large-ll60k"
elif(opt.model_name == "Hubert" and opt.model_size == 'base'):
    model_name = "facebook/hubert-base-ls960"
elif(opt.model_name == "WavLM" and opt.model_size == 'large'):
    model_name = "microsoft/wavlm-large"
elif(opt.model_name == "WavLM" and opt.model_size == 'base'):
    model_name = "microsoft/wavlm-base"
else:
    assert False, "No model found"

subprocess.call(f'rm -rf {opt.target_dir}', shell=True)
subprocess.call(f'mkdir {opt.target_dir}', shell=True)
print(f"model selected: {model_name}")
# model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"#"facebook/wav2vec2-base-960h"#"facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

def create_audio_emds(model, feature_extractor, in_path = '/mnt/c/Users/PCM/Dropbox/span/sub006/2drt/audio', out_path = './datasets/audios', sample_per_sub = 28):
    aud_list = glob.glob(f'{in_path}/*')
    for path in aud_list:
        name = path.split('/')[-1].split('.')[0]
        input_audio, sample_rate = librosa.load(f"{in_path}/{name}.wav",  sr=16000)

        i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            o = model(i.input_values)
        torch.save(o.last_hidden_state, f'{out_path}/{name}.pt')

subjects = glob.glob('/mnt/c/Users/PCM/Dropbox/span/sub*')
for sub in subjects:
    print(sub)
    create_audio_emds(model, feature_extractor, in_path = f'{sub}/2drt/audio', out_path = f'{opt.target_dir}')