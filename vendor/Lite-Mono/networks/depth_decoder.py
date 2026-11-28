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
    stem_ch > 0 (option --hires_stem_ch, "A-lite"): the 48-channel stem feature is first compressed
    to stem_ch channels by a 1x1 conv + ELU, which removes most of the extra MACs of the 3x3 fusion
    conv at 1/2 resolution. stem_ch = 0 keeps the full 48 channels (original --hires_decoder).
    upsample_head (option --hires_upsample): "deconv" = ConvTranspose2d(c0,16,2,2)+ELU+Conv3x3 at full
    resolution (original --hires_decoder); "subpixel" = Conv3x3(c0, 4) at 1/2 resolution + PixelShuffle(2),
    i.e. the sub-pixel convolution of SuperDepth (Pillai et al., ICRA 2019), much cheaper in MACs.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, num_ch_stem=48,
                 stem_ch=0, upsample_head="deconv"):
        super().__init__(num_ch_enc, scales, num_output_channels, use_skips)
        c0 = int(self.num_ch_dec[0])
        self.stem_ch = stem_ch
        skip_ch = stem_ch if stem_ch > 0 else num_ch_stem
        self.convs[("upconv", 0, 1)] = ConvBlock(c0 + skip_ch, c0)
        assert upsample_head in ("deconv", "subpixel"), upsample_head
        if 0 in self.scales and upsample_head == "subpixel":
            self.convs[("dispconv", 0)] = nn.Sequential(
                Conv3x3(c0, 4 * num_output_channels),
                nn.PixelShuffle(2))
        elif 0 in self.scales:
            self.convs[("dispconv", 0)] = nn.Sequential(
                nn.ConvTranspose2d(c0, 16, kernel_size=2, stride=2),
                nn.ELU(inplace=True),
                Conv3x3(16, num_output_channels))
        if stem_ch > 0:
            self.convs[("stemproj",)] = nn.Sequential(nn.Conv2d(num_ch_stem, stem_ch, 1), nn.ELU(inplace=True))
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.apply(self._init_weights)

    def forward(self, input_features):
        stem, input_features = input_features[0], input_features[1:]
        if self.stem_ch > 0:
            stem = self.convs[("stemproj",)](stem)
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


class _Residual(nn.Module):
    """ReLU-Conv3x3-ReLU-Conv3x3 residual unit (DPT / Depth Pro 2 FeatureFusionBlock2d)."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, dim, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(dim, dim, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class _Fusion(nn.Module):
    """DPT feature fusion as in Depth Pro 2: add refined skip, refine, learned 2x upsampling, 1x1 conv."""

    def __init__(self, dim, with_skip):
        super().__init__()
        self.resnet1 = _Residual(dim) if with_skip else None
        self.resnet2 = _Residual(dim)
        self.deconv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.resnet1(skip)
        return self.out_conv(self.deconv(self.resnet2(x)))


class DPTDepthDecoder(nn.Module):
    """Local variant (option --dpt_decoder), not in the Lite-Mono paper.

    Decoder structure of Depth Pro 2 (third_party/ml-depth-pro-2, network/decoder.py: DPTDecoder,
    FeatureFusionBlock2d, create_inverse_depth_head), i.e. the DPT decoder (Ranftl et al., ICCV 2021),
    on the unchanged Lite-Mono encoder (features at 1/4, 1/8, 1/16):
      - every encoder level is projected to `dim` channels by a 3x3 conv;
      - every stage refines with residual units and upsamples 2x with a learned transposed conv
        (the original decoder upsamples every stage bilinearly);
      - the full-resolution disparity comes from the Depth Pro 2 head: 3x3 conv at 1/2, learned 2x
        transposed conv, 3x3 conv to head_ch, ReLU, 1x1 conv.
    Kept from Lite-Mono so that training and evaluation are unchanged: sigmoid disparity output
    (Depth Pro 2 uses ReLU on metric inverse depth) and the multi-scale outputs for the
    self-supervised loss: ("disp", 2) at 1/4 and ("disp", 1) at 1/2 from a 3x3 conv on the stage
    output, ("disp", 0) at full resolution from the head.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, dim=64, head_ch=32):
        super().__init__()
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.dim = dim
        self.convs = OrderedDict()
        for i, c in enumerate(num_ch_enc):
            self.convs[("proj", i)] = nn.Conv2d(int(c), dim, 3, padding=1, bias=False)
        n = len(num_ch_enc)
        for i in range(n):
            self.convs[("fusion", i)] = _Fusion(dim, with_skip=i < n - 1)
        # stage outputs: fusion 1 -> 1/4 (scale 2), fusion 0 -> 1/2 (scale 1)
        for s in self.scales:
            if s in (1, 2):
                self.convs[("dispconv", s)] = Conv3x3(dim, num_output_channels)
        self.convs[("head",)] = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=2, stride=2),
            nn.Conv2d(dim // 2, head_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_ch, num_output_channels, 1))
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
        n = len(input_features)
        x = self.convs[("fusion", n - 1)](self.convs[("proj", n - 1)](input_features[-1]))
        for i in range(n - 2, -1, -1):
            x = self.convs[("fusion", i)](x, self.convs[("proj", i)](input_features[i]))
            s = i + 1  # fusion 1 output is 1/4 -> scale 2, fusion 0 output is 1/2 -> scale 1
            if s in self.scales and ("dispconv", s) in self.convs:
                self.outputs[("disp", s)] = self.sigmoid(self.convs[("dispconv", s)](x))
        if 0 in self.scales:
            self.outputs[("disp", 0)] = self.sigmoid(self.convs[("head",)](x))
        return self.outputs
