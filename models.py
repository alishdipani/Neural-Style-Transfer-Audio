from turtle import forward
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNNModel(nn.Module):
    def __init__(self) -> None:
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = out.view(out.size(0), -1)
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
    def __init__(self, target, weight) -> None:
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
    
    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output
    
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss