
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class multilingualBERT_Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states

class DistilBERT_Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states
    
class TinyBERT_Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states
    
class MobileBERT_Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        self.model = AutoModel.from_pretrained("google/mobilebert-uncased").to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states

class MPNet_Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        self.model = AutoModel.from_pretrained("microsoft/mpnet-base").to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states
    
class Tokenizer():
    def __init__(self, device, max_length=30):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.max_length = max_length
        self.device = device
        
    def __call__(self, annotations):
        inputs = self.tokenizer(annotations, padding='max_length', return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        return hidden_states
    
class Text_Encoder(nn.Module):
    def __init__(self, max_length=30, in_dim = 768):
        super(Text_Encoder, self).__init__()
        self.emb = nn.Sequential(        
            nn.Linear(max_length * in_dim, 1024),
            nn.Linear(1024 , 512),
            nn.Tanh()
        )

    def forward(self, x):
        return self.emb(x.view(x.shape[0], -1))
    
class Image_Encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Image_Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*4, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*8, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*8, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*8, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*8, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.Tanh()
        )
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU(),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        y = self.model(x)        
        return y.view(y.size(0), -1)
    


if __name__ == "__main__":
    text =['The noisy texture in this picture doesn’t work for me. Can you smooth it out?', 'Can you help sharpen this blurry image and make it clear?', 'The foggy appearance is m the subject. Can you make it sharper?', 'The noisy texture in this picture doesn’t work for me. Can you smooth it out?', 'Can you help sharpen this blurry image and make it clear?', 'The foggy appearance is making the photo look low quality. Can you fix it?', 'This image has too much grain for my liking. Can you make it look more professional?', 'The photo is too grainy for printing. Please clean it up so it looks better.', 'Can you enhance the visibility in this dim photo and make it clearer?', 'This photo is too noisy to use in my presentation. Can you enhance it?', 'Can you fix this blurry image and make it more detailed?', 'Can you improve the exposure and details in this dark image?', 'This image has noticeable grain in the shadows. Can you clean it up?', 'Could you improve the low-light areas of this photo and make it clearer?', 'The photo is blurry. Can you enhance its clarity?', 'This picture has too much haze in the foreground. Could you clean it up?']
    print(len(text))
    T = TinyBERT_Tokenizer('cuda')
    p = T(text)
    print(p.shape)

    # # model = Text_Encoder(in_dim=512).to('cuda')
    # # pred = model(p)
    # # print(pred.shape)
    # x = torch.randn((1, 3, 256, 256))
    # model = Image_Encoder(in_channels=3, features=64)
    # y = model(x)
    # print(y.shape)

