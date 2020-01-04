import torch

from torch import nn
from torchvision.models import resnet34


class DPOD(nn.Module):

    def __init__(self, pretrained=True, num_classes=79, num_colors=256, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-3])
        intermediate_activations = []
        self.intermediate_activations = intermediate_activations

        for n, c in self.encoder.named_children():
            print(n, c)

        def hook(module, ins, outs):
            intermediate_activations.append(outs)

        self.encoder[2].register_forward_hook(hook)
        self.encoder[4].register_forward_hook(hook)
        self.encoder[5].register_forward_hook(hook)

        self.class_head = DecoderHead(num_classes, **kwargs)
        self.u_head = DecoderHead(num_colors, **kwargs)
        self.v_head = DecoderHead(num_colors, **kwargs)

    def forward(self, ins):
        self.intermediate_activations.clear()
        print(len(self.intermediate_activations))
        features = self.encoder(ins)
        print("main fwd", features.shape)
        for ac in self.intermediate_activations:
            print(ac.shape)
        classes = self.class_head(features, self.intermediate_activations)
        u_channel = self.u_head(features, self.intermediate_activations)
        v_channel = self.v_head(features, self.intermediate_activations)
        return classes, u_channel, v_channel


class DecoderHead(nn.Module):
    def __init__(self, num_classes, inter_sizes=[(106, 85), (212, 170), (423, 339)], output_size=(846, 677)):
        """
            Default sizes are aligned to match resnet-34 activations, when size of the input image is (3384//4, 2710//4)
        Args:
            num_classes: number of classification channels
            inter_sizes: sizes of intermediate feature maps from encoder architecture
            output_size: image size
        """
        super().__init__()
        self.ups_layers = [
            nn.Upsample(size=inter_sizes[0], mode='bilinear'),
            nn.Upsample(size=inter_sizes[1], mode='bilinear'),
            nn.Upsample(size=inter_sizes[2], mode='bilinear'),
            nn.Upsample(size=output_size, mode='bilinear'),
        ]
        self.conv_layers = [
            nn.Conv2d(256 + 128, 128, kernel_size=(3, 3), padding=(1, 1)),  # padding
            nn.Conv2d(128 + 64, 64, kernel_size=(3, 3), padding=(1, 1)),  # padding
            nn.Conv2d(64 + 64, 64, kernel_size=(3, 3), padding=(1, 1)),  # padding
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),  # padding
            nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1, 1)),  # padding
        ]

    def forward(self, features, intermediate):
        for ups, inter, conv in zip(self.ups_layers[:3], intermediate[::-1], self.conv_layers[:3]):
            print(features.shape)
            features1 = ups(features)
            print(features1.shape, inter.shape)
            features2 = torch.cat([features1, inter], dim=1)
            features = conv(features2)

        features = self.ups_layers[3](features)
        features = self.conv_layers[3](features)
        features = self.conv_layers[4](features)
        return features

