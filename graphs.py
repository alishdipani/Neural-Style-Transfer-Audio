import librosa
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

script, content_audio_name, style_audio_name, output_audio_name = argv

N_FFT=2048
def read_audio_spectum(filename):
	x, fs = librosa.load(filename, duration=58.04) # Duration=58.05 so as to make sizes convenient
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)
	S = np.log1p(np.abs(S))  
	return S, fs

style_audio, style_sr = read_audio_spectum(style_audio_name)
content_audio, content_sr = read_audio_spectum(content_audio_name)
output_audio, output_sr = read_audio_spectum(output_audio_name)

print(style_audio.shape)
print(content_audio.shape)
print(output_audio.shape)

plt.figure(figsize=(15,25))
plt.subplot(1,3,1)
plt.title('Content')
plt.imshow(content_audio[:500,:500])
plt.subplot(1,3,2)
plt.title('Style')
plt.imshow(style_audio[:500,:500])
plt.subplot(1,3,3)
plt.title('Result')
plt.imshow(output_audio[:500,:500])
plt.show()