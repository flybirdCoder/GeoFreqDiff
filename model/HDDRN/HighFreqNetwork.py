import torch
from torch import nn

from model.HDDRN.SFCRM import SFCRM


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class HighFreqNetwork(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 dim=48,
                 bias=False,
                 ):
        super(HighFreqNetwork, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoder_level1 = SFCRM(nChannels=dim)

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = SFCRM(nChannels=dim * 2 ** 1)

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.latent = SFCRM(nChannels=dim * 2 ** 2)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = SFCRM(nChannels=dim * 2 ** 1)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = SFCRM(nChannels=dim * 2 ** 1)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input_image):
        inp_enc_level1 = self.patch_embed(input_image)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
