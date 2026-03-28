import os
import librosa
import soundfile as sf
import time

import torch

from models.DCCRN import dccrn

import numpy as np

from config import Config
opt = Config

# when denoising, use cpu
def denoise(mode, speech_file, save_dir, pth=None):
    assert os.path.exists(speech_file), 'speech file does not exist!'

    assert speech_file.endswith('.wav'), 'non-supported speech format!'

    if not os.path.exists(save_dir):
        print('warning: save directory does not exist, it will be created automatically!')
        os.makedirs(save_dir)

    model = dccrn(mode)
    if pth is not None:
        model.load_state_dict(torch.load(pth), strict=True)

    noisy_wav, _ = librosa.load(speech_file, sr=16000)

    noisy_wav = torch.Tensor(noisy_wav).reshape(1, -1)

    torch.cuda.synchronize()
    start = time.time()

    _, denoised_wav = model(noisy_wav)

    torch.cuda.synchronize()
    end = time.time()

    print('process time {0}s on device {1}'.format(end - start, 'cpu'))

    speech_name = os.path.basename(speech_file)[:-4]

    noisy_path = os.path.join(save_dir, speech_name + '_' + 'noisy' + '.wav')
    denoised_path = os.path.join(save_dir, speech_name + '_' + 'denoised' + '.wav')

    noisy_wav = noisy_wav.data.numpy().flatten()
    denoised_wav = denoised_wav.data.numpy().flatten()

    sf.write(noisy_path, noisy_wav, 16000)
    sf.write(denoised_path, denoised_wav, 16000)

def process_predict(wav_path):
    def audioread(path):
        data, fs = sf.read(path)
        if len(data.shape) > 1:
            data = data[0]
        return data,fs
    data,fs = audioread(wav_path)
    return data,fs

def predict_one_wav(wav_path,use_cuda = True):
    args = Config()
    model = dccrn(args.mode)
    model.load_state_dict(torch.load(args.best_path), strict=True)
    audio,nsamples = process_predict(wav_path)
    inputs = torch.from_numpy(audio).unsqueeze(dim=0).to(dtype=torch.float32)
    _, denoised_wav = model(inputs)
    sf.write(os.path.join(os.path.dirname(wav_path),"denoised_wav_"+os.path.basename(wav_path)) , denoised_wav[0].detach().numpy(), 16000)

    ##### 流式推理
    audio = np.pad(audio, (100, 100), 'constant')
    audio_length = len(audio)
    hop = 100
    stride = int((audio_length - hop) / hop)

    audio_enhance = []
    for i in range(stride):
        input = audio[hop*i:hop*(i+1)]
        input = torch.from_numpy(input).unsqueeze(dim=0).to(dtype=torch.float32)
        output_stride = model.forward_streaming(input).detach().numpy().tolist()
        audio_enhance+=output_stride

    sf.write(os.path.join(os.path.dirname(wav_path), "denoised_streaming_" + os.path.basename(wav_path)), audio_enhance, 16000)
    return None



if __name__ == '__main__':
    wave_test = 'assert/D4_750_cafe.wav'
    predict_one_wav(wave_test)



    # https://www.skyreels.ai/dev/api-keys
    # sk_fa25ebd36dec45169bd2f754bbbb3f90

