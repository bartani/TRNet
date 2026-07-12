import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()
        features = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features

        self.resize_input = resize_input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))
        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(
                pre, pos), torch.nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
                [9, 10, 11], [12, 13], [14, 15], [16, 17],
                [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        if self.resize_input:
            x = F.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key, value) in list(zip(self.relus, features))}
        return out
    
class L1(): 
    def __init__(self,device):
        self.calc = torch.nn.L1Loss().to(device)
    
    def __call__(self, x, y):
        return self.calc(x, y)
    
class Perceptual():
    def __init__(self, device, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        self.vgg = VGG19().to(device)
        self.criterion = torch.nn.L1Loss().to(device)
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Euclidean distance between anchor-positive and anchor-negative
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Cosine similarities: higher is more similar
        sim_ap = F.cosine_similarity(anchor, positive, dim=1)
        sim_an = F.cosine_similarity(anchor, negative, dim=1)

        # Loss: encourage sim_ap > sim_an + margin
        loss = F.relu(sim_an - sim_ap + self.margin)
        return loss.mean()