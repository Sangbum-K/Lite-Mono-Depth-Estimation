from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs



class HiResDepthDecoder(DepthDecoder):
    """Local variant (option --hires_decoder), not in the Lite-Mono paper.

    Two changes to the last decoder stage, everything else is identical to DepthDecoder:
      1. skip connection from the encoder's 1/2-resolution stem feature (discarded by the
         original decoder, whose finest skip is 1/4 resolution);
      2. the full-resolution disparity (scale 0) comes from a learned 2x transposed-conv
         head instead of a 3x3 conv at 1/2 resolution followed by bilinear upsampling.
    Idea adapted from the learned (transposed-conv) upsampling of DPT / Depth Pro / Depth Pro 2
    and the high-resolution skip fusion of HR-Depth (Lyu et al., AAAI 2021).
    Input: [stem_half, f_1/4, f_1/8, f_1/16] from LiteMono(return_stem=True).
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, num_ch_stem=48):
        super().__init__(num_ch_enc, scales, num_output_channels, use_skips)
        c0 = int(self.num_ch_dec[0])
        self.convs[("upconv", 0, 1)] = ConvBlock(c0 + num_ch_stem, c0)
        if 0 in self.scales:
            self.convs[("dispconv", 0)] = nn.Sequential(
                nn.ConvTranspose2d(c0, 16, kernel_size=2, stride=2),
                nn.ELU(inplace=True),
                Conv3x3(16, num_output_channels))
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.apply(self._init_weights)

    def forward(self, input_features):
        stem, input_features = input_features[0], input_features[1:]
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            if i == 0:
                x += [stem]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if i == 0:
                    f = self.convs[("dispconv", 0)](x)  # learned 2x upsampling to full resolution
                else:
                    f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)
        return self.outputs
