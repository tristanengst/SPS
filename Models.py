

def get_activation(act=leakyrelu):
    """
    """
    if act == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation {act}")

class MLP(nn.Module):

    def __init__(in_dim, out_dim, h_dim=512, n_layers=2, act="leakyrelu"):
        super(MLP, self).__init__()

        if n_layers == 1:
            self.model = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, h_dim)]

            for _ in range(n_layers - 2):
                layers.append(get_activation(act))
                layers.append(nn.Linear(h_dim, h_dim))

            layers.append(get_activation(act))
            layern.append(nn.Linear(h_dim, out_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)

class FeatBlock(nn.Module):
    """A block for feature extraction."""
    def __init__(self, in_shape, out_shape):



class SPSBlock(nn.Module):
    """Returns a (hidden_representation, semantic_representation) tuple.
    """
    def __init__(self, feat_block):
        super(SPSBlock, self.__init__(self))
        self.feat_block = feat_block
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.semantic_mlp = MLP()

    def forward(self, x):
        feat = self.feat_block(x)
        return feat, self.semantic_mlp(self.avg_pool(feat))

class SPSNet(nn.Module):
    """Output layers for a feature extractor used for SPS.
    """
    def __init__(self, feature_extractor_input, feature_extractor_blocks):
        super(SPSLayers, self).__init__()

        self.input = feature_extractor_input
        self.feature_extractor_blocks = nn.ModuleList(*feature_extractor_blocks)

    def forward(self, x):
        fb = self.input(x)
        semantic_feats = []

        for b,m in zip(self.feature_extractor_blocks, self.mlps):
            fb = b(fb)
            semantic_feats.append(m(fb))

        semantic_feats = torch.cat([l.flatten(start_dim=1) for l in feats], axis=1)

        return self.projection(semantic_feats)


class SPSResNet(nn.Module):

    def __init__(self, backbone="resnet18"):
        if backbone == "resnet18":
            self.backbone = models.resnet18()
        else:
            raise ValueError(f"Unknown backbone '{backbone}'")

        extracted_modules = ["layer1", "layer2", "layer3", "layer4", "fc", "avgpool"]



    def forward(self, x):
