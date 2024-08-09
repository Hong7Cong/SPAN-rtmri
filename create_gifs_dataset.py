
from PIL import Image
from torch.utils.data import Dataset
import glob, torch
import subprocess
from utils import *
from PIL import Image, ImageSequence
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path2span", type=str, default='/mnt/c/Users/PCM/Dropbox/span', help='where the original span dataset located')
parser.add_argument("--target_dir", type=str, default='./datasets/preprocessed_dataset', help='destination to save preprocessed dataset')
opt = parser.parse_args()
print(opt)

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

target_dir = opt.target_dir
# subprocess.call(f'rm -rf {opt.target_dir}', shell=True)
# subprocess.call(f'mkdir {opt.target_dir}', shell=True)
subprocess.call(f'rm -rf {opt.target_dir}/train', shell=True)
subprocess.call(f'rm -rf {opt.target_dir}/test', shell=True)
subprocess.call(f'mkdir {opt.target_dir}/train', shell=True)
subprocess.call(f'mkdir {opt.target_dir}/test', shell=True)
# subprocess.call(f'mkdir {opt.target_dir}/train/gifs', shell=True)
# subprocess.call(f'mkdir {opt.target_dir}/test/gifs', shell=True)
# subprocess.call(f'mkdir {opt.target_dir}/train/audio_emb', shell=True)
# subprocess.call(f'mkdir {opt.target_dir}/test/audio_emb', shell=True)

# MAKE TRAINING DATASET
subjects = glob.glob(f'{opt.path2span}/sub*')[:60]
for sub in subjects:
    vids = glob.glob(f'{sub}/2drt/video/*')[:28]
    # vids = glob.glob(f'/mnt/c/Users/PCM/Dropbox/span/sub006/2drt/video/*')
    window = 0.2 # step = window - overlap
    overlap = 0
    for i in range(len(vids)):
        for skip in np.arange(0, int(get_length(vids[i]))-1, window-overlap):
            command = f"ffmpeg -y -ss {skip} -t {window} -i {vids[i]} -vf \"fps=50,scale=64:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {target_dir}/train/{vids[i].split('/')[-1].split('.')[0]}-{int(skip*50)}.gif"
            subprocess.call(command, shell=True)

# MAKE TEST DATASET
subjects = glob.glob(f'{opt.path2span}/sub*')[60:]
for sub in subjects:
    vids = glob.glob(f'{sub}/2drt/video/*')
    window = 0.2 # step = window - overlap
    overlap = 0
    for i in range(len(vids)):
        for skip in np.arange(0, int(get_length(vids[i]))-1, window-overlap):
            command = f"ffmpeg -y -ss {skip} -t {window} -i {vids[i]} -vf \"fps=50,scale=64:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {target_dir}/test/{vids[i].split('/')[-1].split('.')[0]}-{int(skip*50)}.gif"
            subprocess.call(command, shell=True)

subjects = glob.glob(f'{opt.path2span}/sub*')[:60]
for sub in subjects:
    vids = glob.glob(f'{sub}/2drt/video/*')[28:]
    window = 0.2 # step = window - overlap
    overlap = 0
    for i in range(len(vids)):
        for skip in np.arange(0, int(get_length(vids[i]))-1, window-overlap):
            command = f"ffmpeg -y -ss {skip} -t {window} -i {vids[i]} -vf \"fps=50,scale=64:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {target_dir}/test/{vids[i].split('/')[-1].split('.')[0]}-{int(skip*50)}.gif"
            subprocess.call(command, shell=True)