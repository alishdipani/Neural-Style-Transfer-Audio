from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 
from sys import argv
import torchvision.transforms as transforms
import copy
import librosa

class CNNModel(nn.Module):
		def __init__(self):
			super(CNNModel, self).__init__()
			self.cnn1 = nn.Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
			#self.nl1 = nn.ReLU()
			#self.pool1 = nn.AvgPool1d(kernel_size=5)
			#self.fc1 = nn.Linear(4096*2500,2**5)
			#self.nl3 = nn.ReLU()
			#self.fc2 = nn.Linear(2**10,2**5)
		
		def forward(self, x):
			out = self.cnn1(x)
			#out = self.nl1(out)
			#out = self.pool1(out)
			out = out.view(out.size(0),-1)
			#out = self.fc1(out)
			#out = self.nl3(out)
			#out = self.fc2(out)
			return out


class GramMatrix(nn.Module):

	def forward(self, input):
		a, b, c = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
		features = input.view(a * b, c)  # resise F_XL into \hat F_XL
		G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
		return G.div(a * b * c)


class StyleLoss(nn.Module):

	def __init__(self, target, weight):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()

	def forward(self, input):
		self.output = input.clone()
		self.G = self.gram(input)
		self.G.mul_(self.weight)
		self.loss = self.criterion(self.G, self.target)
		return self.output

	def backward(self,retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss

if __name__ == '__main__':
	#print('Enter the names of SCRIPT, Content audio, Style audio')
	script, content_audio_name , style_audio_name = argv

	# USING LIBROSA
	N_FFT=2048
	def read_audio_spectum(filename):
		x, fs = librosa.load(filename, duration=58.04) # Duration=58.05 so as to make sizes convenient
		S = librosa.stft(x, N_FFT)
		p = np.angle(S)
		S = np.log1p(np.abs(S))  
		return S, fs

	style_audio, style_sr = read_audio_spectum(style_audio_name)
	content_audio, content_sr = read_audio_spectum(content_audio_name)

	if(content_sr == style_sr):
		print('Sampling Rates are same')
	else:
		print('Sampling rates are not same')
		exit()

	num_samples=style_audio.shape[1]	
		
	style_audio = style_audio.reshape([1,1025,num_samples])
	content_audio = content_audio.reshape([1,1025,num_samples])


	if torch.cuda.is_available():
		style_float = Variable((torch.from_numpy(style_audio)).cuda())
		content_float = Variable((torch.from_numpy(content_audio)).cuda())	
		print('using CUDA')
	else:
		style_float = Variable(torch.from_numpy(style_audio))
		content_float = Variable(torch.from_numpy(content_audio))
		print('using CPU')
	#style_float = style_float.unsqueeze(0)
	
	#style_float = style_float.view([1025,1,2500])
	
	'''
	print(style_float.size())
	exit()
	'''
	#style_float = style_float.unsqueeze(0)
	#content_float = content_float.unsqueeze(0)
	#content_float = content_float.reshape(1025,1,2500)
	
	#content_float = content_float.unsqueeze(0)
	#content_float = content_float.squeeze(0)

	cnn = CNNModel()
	if torch.cuda.is_available():
		cnn = cnn.cuda()
	style_layers_default = ['conv_1']

	style_weight=2500

	def get_style_model_and_losses(cnn, style_float,style_weight=style_weight, style_layers=style_layers_default): #STYLE WEIGHT
		
		cnn = copy.deepcopy(cnn)
		style_losses = []
		model = nn.Sequential()  # the new Sequential module network
		gram = GramMatrix()  # we need a gram module in order to compute style targets
		if torch.cuda.is_available():
			model = model.cuda()
			gram = gram.cuda()

		name = 'conv_1'
		model.add_module(name, cnn.cnn1)
		if name in style_layers:
			target_feature = model(style_float).clone()
			target_feature_gram = gram(target_feature)
			style_loss = StyleLoss(target_feature_gram, style_weight)
			model.add_module("style_loss_1", style_loss)
			style_losses.append(style_loss)

		#name = 'pool_1'
		#model.add_module(name, cnn.pool1)

		'''name = 'fc_1'
		model.add_module(name, cnn.fc1)

		name = 'nl_9'
		model.add_module(name, cnn.nl9)

		name = 'fc_2'
		model.add_module(name, cnn.fc2)'''

		return model, style_losses


	input_float = content_float.clone()
	#input_float = Variable(torch.randn(content_float.size())).type(torch.FloatTensor)

	learning_rate_initial = 0.03

	def get_input_param_optimizer(input_float):
		input_param = nn.Parameter(input_float.data)
		#optimizer = optim.Adagrad([input_param], lr=learning_rate_initial, lr_decay=0.0001,weight_decay=0)
		optimizer = optim.Adam([input_param], lr=learning_rate_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
		return input_param, optimizer

	num_steps= 2500

	def run_style_transfer(cnn, style_float, input_float, num_steps=num_steps, style_weight=style_weight): #STYLE WEIGHT, NUM_STEPS
		print('Building the style transfer model..')
		model, style_losses= get_style_model_and_losses(cnn, style_float, style_weight)
		input_param, optimizer = get_input_param_optimizer(input_float)
		print('Optimizing..')
		run = [0]

		while run[0] <= num_steps:
			def closure():
            	# correct the values of updated input image
				input_param.data.clamp_(0, 1)

				optimizer.zero_grad()
				model(input_param)
				style_score = 0

				for sl in style_losses:
					#print('sl is ',sl,' style loss is ',style_score)
					style_score += sl.backward()

				run[0] += 1
				if run[0] % 100 == 0:
					print("run {}:".format(run))
					print('Style Loss : {:8f}'.format(style_score.item())) #CHANGE 4->8 
					print()

				return style_score


			optimizer.step(closure)
		input_param.data.clamp_(0, 1)
		return input_param.data
		
	output = run_style_transfer(cnn, style_float, input_float)
	if torch.cuda.is_available():
		output = output.cpu()

	#output = output.squeeze(0)
	output = output.squeeze(0)
	output = output.numpy()
	#print(output.shape)
	#output = output.resize([1025,2500])
	
	N_FFT=2048
	a = np.zeros_like(output)
	a = np.exp(output) - 1

	# This code is supposed to do phase reconstruction
	p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
	for i in range(500):
		S = a * np.exp(1j*p)
		x = librosa.istft(S)
		p = np.angle(librosa.stft(x, N_FFT))

	OUTPUT_FILENAME = 'output1D_4096_iter'+str(num_steps)+'_c'+content_audio_name+'_s'+style_audio_name+'_sw'+str(style_weight)+'_k3s1p1.wav'
	librosa.output.write_wav(OUTPUT_FILENAME, x, style_sr)

	print('DONE...')
