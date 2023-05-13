import os, sys
from libs import *

class PretextsCA(nn.Module):
    def __init__(self, 
        num_classes = 0, 
    ):
        super(PretextsCA, self).__init__()
        self.pretext_resnet50_simclr = self.load_pretext("resnet50_simclr")
        self.pretext_resnet50_mocov2 = self.load_pretext("resnet50_mocov2")
        self.mha_resnet50_simclr = nn.MultiheadAttention(
            embed_dim = 300, 
            num_heads = 2, dropout = 0.2, 
            batch_first = True, 
        )
        self.mha_resnet50_mocov2 = nn.MultiheadAttention(
            embed_dim = 300, 
            num_heads = 2, dropout = 0.2, 
            batch_first = True, 
        )

        self.classifier = nn.Linear(
            300, num_classes, 
        )

    def load_pretext(self, 
        state_dict_path, 
    ):
        if "dino" not in state_dict_path:
            state_dict = torch.load(
                "../../pretexts/" + state_dict_path, map_location = "cpu", 
            )["classy_state_dict"]["base_model"]["model"]["trunk"]
            state_dict = collections.OrderedDict([
                (key.replace("_feature_blocks.", ""), value) if "_feature_blocks." in key else (key, value) for key, value in state_dict.items()
            ])

            pretext = torchvision.models.__dict__["resnet50"](); pretext.fc = nn.Identity(); 
            pretext.load_state_dict(
                state_dict = state_dict, strict = True, 
            )
            pretext.fc = nn.Linear(
                2048, 300, 
            )
        else:
            if "dinov2" in state_dict_path:
                pretext = torch.hub.load(
                    "facebookresearch/dinov2", state_dict_path, 
                )
            else:
                pretext = torch.hub.load(
                    "facebookresearch/dino:main", state_dict_path, 
                )
            pretext.head = nn.Linear(
                384, 300, 
            )

        return pretext

    def forward(self, 
        input, 
    ):
        feature_resnet50_simclr = self.pretext_resnet50_simclr(input)
        feature_resnet50_mocov2 = self.pretext_resnet50_mocov2(input)
        feature = torch.mean(
            torch.stack(
                [
                    feature_resnet50_simclr, 
                    feature_resnet50_mocov2, 
                ]
            ), 
            dim = 0, 
        )
        attn_feature_resnet50_simclr = feature_resnet50_simclr + self.mha_resnet50_simclr(
            feature_resnet50_simclr, 
            feature_resnet50_mocov2, feature_resnet50_mocov2, 
        )[0]
        attn_feature_resnet50_mocov2 = feature_resnet50_mocov2 + self.mha_resnet50_mocov2(
            feature_resnet50_mocov2, 
            feature_resnet50_simclr, feature_resnet50_simclr, 
        )[0]
        attn_feature = torch.mean(
            torch.stack(
                [
                    attn_feature_resnet50_simclr, 
                    attn_feature_resnet50_mocov2, 
                ]
            ), 
            dim = 0, 
        )

        output = self.classifier(attn_feature)

        return (
            feature, 
            attn_feature, 
        ), output