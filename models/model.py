import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101


class DetectionModel(nn.Module):
    """
    Hybrid Model from Tiny Faces paper
    """

    def __init__(self, base_model=resnet101, num_templates=1, num_objects=1):
        super().__init__()
        # 4 is for the bounding box offsets
        output = (num_objects + 4)*num_templates
        self.model = base_model(pretrained=True)

        # delete unneeded layer
        del self.model.layer4

        self.score_res3 = nn.Conv2d(in_channels=512, out_channels=output,
                                    kernel_size=1, padding=0)
        self.score_res4 = nn.Conv2d(in_channels=1024, out_channels=output,
                                    kernel_size=1, padding=0)

    def learnable_parameters(self, lr):
        parameters = [
            # Be T'Challa. Don't freeze.
            {'params': self.model.parameters(), 'lr': lr},
            {'params': self.score_res3.parameters(), 'lr': 0.1*lr},
            {'params': self.score_res4.parameters(), 'lr': 1*lr},
        ]
        return parameters

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        # res2 = x

        x = self.model.layer2(x)
        res3 = x

        x = self.model.layer3(x)
        res4 = x

        score_res3 = self.score_res3(res3)

        score_res4 = self.score_res4(res4)
        # we use align_corners=False since its behavior matches with OpenCV resize
        score4 = F.interpolate(score_res4, size=score_res3.shape[2:4],
                               mode='bilinear', align_corners=False)

        score = score_res3 + score4

        return score
