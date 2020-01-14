import torch

from torch import nn
from torchvision.models import resnet34, resnet18


class DPOD(nn.Module):

    def __init__(self, pretrained=True, num_classes=79+1, num_colors=256, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-3])
        
        # Freeze first five layers
        ct = 0
        for _, c in self.encoder.named_children():
            if ct < 5:
                for param in c.parameters():
                    param.requires_grad = False
            ct += 1

        intermediate_activations = []
        self.intermediate_activations = intermediate_activations

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
        features = self.encoder(ins)
        classes = self.class_head(features, self.intermediate_activations)
        u_channel = self.u_head(features, self.intermediate_activations)
        v_channel = self.v_head(features, self.intermediate_activations)
        return classes, u_channel, v_channel


class DecoderHead(nn.Module):
    def __init__(self, num_classes, image_size=(846, 677)):
        """
            Default sizes are aligned to match resnet-34 activations, when size of the input image is (3384//4, 2710//4)
        Args:
            num_classes: number of classification channels
            image_size: image size
        """
        super().__init__()
        
        self.inter_sizes = []
        w = image_size[0]
        h = image_size[1]
        for _ in range(3):
            w = (w - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            self.inter_sizes.append((w, h))

        self.inter_sizes = self.inter_sizes[::-1]

        self.ups1 = nn.Upsample(size=self.inter_sizes[0], mode='bilinear')
        self.ups2 = nn.Upsample(size=self.inter_sizes[1], mode='bilinear')
        self.ups3 = nn.Upsample(size=self.inter_sizes[2], mode='bilinear')
        self.ups4 = nn.Upsample(size=image_size, mode='bilinear')
        
        self.conv1 = nn.Conv2d(256 + 128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(128 + 64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=(1, 1))

        self.ups_layers = [self.ups1, self.ups2, self.ups3]
        self.conv_layers = [self.conv1, self.conv2, self.conv3]

    def forward(self, features, intermediate):
        for ups, inter, conv in zip(self.ups_layers, intermediate[::-1], self.conv_layers):
            features1 = ups(features)
            features2 = torch.cat([features1, inter.cuda()], dim=1)
            features = conv(features2.cuda())

        features = self.ups4(features)
        features = self.conv4(features)
        features = self.conv5(features)
        return features


