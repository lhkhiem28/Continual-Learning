import os, sys
from libs import *

class PretextsCA(nn.Module):
    def __init__(self, 
        num_classes = 0, 
    ):
        super(PretextsCA, self).__init__()
        self.resnet50_simclr = self.load_pretext("/home/ubuntu/khiem.lh/Free/Continual-Learning/pretexts/resnet50_simclr")
        self.resnet50_mocov2 = self.load_pretext("/home/ubuntu/khiem.lh/Free/Continual-Learning/pretexts/resnet50_mocov2")
        self.vits16_dino = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", 
            map_location = "cpu", 
        )
        for parameter in self.vits16_dino.parameters():
            parameter.requires_grad = False

    def load_pretext(self, 
        state_dict_path, 
    ):
        state_dict = torch.load(
            state_dict_path, 
            map_location = "cpu", 
        )
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        state_dict = collections.OrderedDict([
            (key.replace("_feature_blocks.", ""), value) if "_feature_blocks." in key else (key, value) 
            for key, value in state_dict.items()
        ])

        pretext = torchvision.models.__dict__["resnet50"]()
        pretext.fc = nn.Identity()
        pretext.load_state_dict(
            state_dict = state_dict, 
            strict = True, 
        )
        for parameter in pretext.parameters():
            parameter.requires_grad = False

        pretext.fc = nn.Linear(
            2048, 384, 
        )

        return pretext