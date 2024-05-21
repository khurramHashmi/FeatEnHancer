from .feat_enhancer import FeatEnHancer

def build_feat_enhancer(in_channels):
    return FeatEnHancer(in_channels)
