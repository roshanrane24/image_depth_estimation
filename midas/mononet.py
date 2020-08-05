import torch
import torch.nn as nn
from .utils import _make_encoder
from .blocks import FeatureFusionBlock, Interpolate

class MidasNet(nn.Module):
    """
    Network for depth estimation
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """
        Init Function
        <<<
        - path:str (optional) > Path to saved model [None].
        - features:int (optional) > Number of features [256].
        - non_negative (optional > to use ReLU or not [True].
        >>>
        """
        super(MidasNet, self).__init__()   
        print("loading Weights...\n", path)

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        
        self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True) if non_negative else nn.Identity())
        
        if path:
            self.load(path)

    def load(self, path):
        """
        Load model
        <<<
        - path:str > path to model file
        >>>
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

    def forward(self, x):
        """
        forward pass of the model.
        <<<
        - x:tensor > Input data (image)
        >>>
        - :tensor > depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return torch.squeeze(out, dim=1)
