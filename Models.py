import torch
import torch.nn as nn
from torchvision import models

def normalize_tensor(x, eps=1e-10):
    """Returns tensor [x] after normalization."""
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)

class VGGFeatures(nn.Module):
    """A network returning the activations of the five Conv2D layers used in
    [Zhang et al., 2018. https://richzhang.github.io/PerceptualSimilarity].

    Args:
    pretrained  -- whether or not the VGG network is pretrained
    """
    def __init__(self, pretrained=False):
        super(VGGFeatures, self).__init__()
        vgg_feat_layers = models.vgg16(pretrained=pretrained).features
        self.slice1 = vgg_feat_layers[:4]
        self.slice2 = vgg_feat_layers[4:9]
        self.slice3 = vgg_feat_layers[9:16]
        self.slice4 = vgg_feat_layers[16:23]
        self.slice5 = vgg_feat_layers[23:30]

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5

class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""
    def __init__(self, chn_in, chn_out=1, dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout()] if dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)

class ScaledVGGFeatures(nn.Module):
    """Neural network returning the concatenation of linear scalings of the
    activations of five Conv2D layers of a VGG16 network, as in
    [Zhang et al., 2018. https://richzhang.github.io/PerceptualSimilarity].

    Args:
    pretrained          -- whether or not the VGG network is pretrained
    already_normalized  -- whether or not inputs are already normalized
    dropout             -- use dropout
    """
    def __init__(self, pretrained_vgg=False, already_normalized=False, dropout=False):
        super(ScaledVGGFeatures, self).__init__()
        self.already_normalized = already_normalized
        self.vgg16 = VGGFeatures(pretrained=pretrained_vgg)
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

        self.lin_layers = nn.ModuleList([
            NetLinLayer(64, dropout=dropout),
            NetLinLayer(128, dropout=dropout),
            NetLinLayer(256, dropout=dropout),
            NetLinLayer(512, dropout=dropout),
            NetLinLayer(512, dropout=dropout),
        ])

    def forward(self, x):
        x = x if self.already_normalized else 2 * x - 1
        x = (x - self.shift) / self.scale
        vgg_feats = [normalize_tensor(f) for f in self.vgg16(x)]
        feats = [l(v) for l,v in zip(self.lin_layers, vgg_feats)]
        return torch.cat([l.flatten(start_dim=1) for l in feats], axis=1)
