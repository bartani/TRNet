import torch
import torch.nn as nn
from thop import profile

class FiLM(nn.Module):
    def __init__(self, text_dim, num_channels):
        super().__init__()
        self.gamma = nn.Linear(text_dim, num_channels)
        self.beta = nn.Linear(text_dim, num_channels)

    def forward(self, x, t):
        gamma = self.gamma(t).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta(t).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        # print("x:", x.shape, "gamma:", gamma.shape, "beta:", beta.shape)
        return gamma * x + beta

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(encoder, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = self.enc_layer(features, features*2)
        self.down2 = self.enc_layer(features*2, features*4)
        self.down3 = self.enc_layer(features*4, features*8)
        self.down4 = self.enc_layer(features*8, features*8)
        self.down5 = self.enc_layer(features*8, features*8)
        self.down6 = self.enc_layer(features*8, features*8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
    
    def enc_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        return d1, d2, d3, d4, d5, d6, d7, bottleneck


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, text_dim=512):
        super().__init__()
        self.enc = encoder(in_channels=3, features=64)
        self.film_512 = FiLM(text_dim, 512)
        self.film_256 = FiLM(text_dim, 256)
        self.film_128 = FiLM(text_dim, 128)
        self.film_64 = FiLM(text_dim, 64)

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, delta):
        d1, d2, d3, d4, d5, d6, d7, bottleneck = self.enc(x)
        bottleneck = self.film_512(bottleneck, delta)
        
        up1 = self.up1(bottleneck)
        up1 = self.film_512(up1, delta)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up2 = self.film_512(up2, delta)
        up3 = self.up3(torch.cat([up2, d6], 1))
        up3 = self.film_512(up3, delta)
        up4 = self.up4(torch.cat([up3, d5], 1))
        up4 = self.film_512(up4, delta)
        up5 = self.up5(torch.cat([up4, d4], 1))
        up5 = self.film_256(up5, delta)
        up6 = self.up6(torch.cat([up5, d3], 1))
        up6 = self.film_128(up6, delta)
        up7 = self.up7(torch.cat([up6, d2], 1))
        up7 = self.film_64(up7, delta)
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    batch_size = 1
    text_dim=512
    x = torch.randn((batch_size, 3, 256, 256))
    delta = torch.randn((batch_size, text_dim))
    model = Generator(in_channels=3, features=64, text_dim=text_dim)
    pred = model(x, delta)
    print(pred.shape)

    
    flops, params = profile(model, inputs=(x, delta))
    print("flops:", flops/1e9)
    print("params:", params/1e6)


if __name__ == "__main__":
    test()
