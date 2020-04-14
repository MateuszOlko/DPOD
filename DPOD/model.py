import torch

from torch import nn
from torchvision.models import resnet34, resnet18

from DPOD.models_handler import ModelsHandler, rotation_matrix_to_kaggle_aw_pitch_roll
from DPOD.ransacs import pnp_ransac_multiple_instance, pnp_ransac_no_class
from scipy.stats import mode


class DPOD(nn.Module):

    def __init__(self, pretrained=True, num_classes=79+1, num_colors=256, image_size=(2710//8, 3384//8)):
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

        self.class_head = DecoderHead(num_classes, image_size)
        self.u_head = DecoderHead(num_colors, image_size)
        self.v_head = DecoderHead(num_colors, image_size)

    def forward(self, ins):
        self.intermediate_activations.clear()
        features = self.encoder(ins)
        classes = self.class_head(features, self.intermediate_activations)
        u_channel = self.u_head(features, self.intermediate_activations)
        v_channel = self.v_head(features, self.intermediate_activations)
        return classes, u_channel, v_channel


class DecoderHead(nn.Module):
    def __init__(self, num_classes, image_size):
        """
            Sizes of intermediate values are computed based on image_size parameter.
        Args:
            num_classes: number of classification channels
            image_size: image size
        """
        super().__init__()
        
        self.inter_sizes = []
        h = image_size[0]
        w = image_size[1]
        for _ in range(3):
            w = (w - 1) // 2 + 1
            h = (h - 1) // 2 + 1
            self.inter_sizes.append((h, w)) 

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


class PoseBlock(nn.Module):
    def __init__(self, kaggle_dataset_dir_path, num_classes_excluding_background=79, downscaling=8, min_inliers=50, no_class=False, **solvePnPRansacKwargs):
        super(PoseBlock, self).__init__()
        self.models_handler = ModelsHandler(kaggle_dataset_dir_path)
        self.num_models = num_classes_excluding_background
        self.downscaling = downscaling
        self.min_inliers = min_inliers
        self.no_class = no_class
        self.solvePnPRansacKwargs = solvePnPRansacKwargs

    def forward(self, classes, u_channel, v_channel):
        """
        does not return tensors as number of detected instances per image may vary

        :param classes:   (h, w) integer-valued tensor,
            best class pixel wise, any value not in {0, ...,self.num_classes_excluding_background}
            will be treated as background
        :param u_channel: (h, w) uint8-valued tensor, best color pixel wise
        :param v_channel: (h, w) uint8-valued tensor, best color pixel wise
        :return:
        [
            [
                model_id,                           int
                ransac_translation_vector,          (3)   float np.array
                ransac_rotation_matrix,             (3,3) float np.array
            ]
            for each instance found by ransac in image
        ]
        """

        c = classes.cpu().numpy()              # best class pixel wise
        u = u_channel.cpu().numpy()              # best color pixel wise
        v = v_channel.cpu().numpy()              # best color pixel wise
        if self.no_class:
            instances = pnp_ransac_no_class(
                c, u, v, self.downscaling, self.models_handler, self.num_models, min_inliers=self.min_inliers, k=5, **self.solvePnPRansacKwargs)  # todo optimize min_inliers
        else:
            instances = pnp_ransac_multiple_instance(
                c, u, v, self.downscaling, self.models_handler, self.num_models, min_inliers=self.min_inliers, **self.solvePnPRansacKwargs)  # todo optimize min_inliers

        output = []  # list of instances
        for success, ransac_rotation_matrix, ransac_translation_vector, pixel_coordinates_of_inliers, model_id in instances:
            output.append(
                [
                    model_id,
                    ransac_translation_vector,
                    ransac_rotation_matrix
                ]
            )
        return output


class Translator(nn.Module):
    def forward(self, batch_of_instances):
        """
        :param batch_of_instances: as returned by PoseBlock.forward(
        :return: prediction strings
        """
        prediction_strings = []
        for instances in batch_of_instances:
            prediction_strings_for_instances = []
            for n_instance, instance in enumerate(instances):
                model_id, translation_vector, rotation_matrix = instance
                kaggle_yaw, kaggle_pitch, kaggle_roll = rotation_matrix_to_kaggle_aw_pitch_roll(rotation_matrix)
                prediction_string_for_single_instance = ' '.join(
                    [
                        str(kaggle_yaw),
                        str(kaggle_pitch),
                        str(kaggle_roll),
                        str(translation_vector[0]),
                        str(translation_vector[1]),
                        str(translation_vector[2]),
                        str(1 - 0.01*n_instance)  # confidence score
                    ]
                )
                prediction_strings_for_instances.append(prediction_string_for_single_instance)
            prediction_string_for_image = ' '.join(prediction_strings_for_instances)
            prediction_strings.append(prediction_string_for_image)

        return prediction_strings


class RefinementNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.first_block_rgb = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:5])
        self.first_block_rendered = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:5])
        self.second_block = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[5:-1])
        self.top_fc = nn.Linear(512, 256)

        self.rot_f1 = nn.Linear(256 + 4, 32)
        self.rot_f2 = nn.Linear(32 + 4, 4)

        self.z_f1 = nn.Linear(256 + 1, 64)
        self.z_f2 = nn.Linear(64 + 1, 1)

        self.xy_f1 = nn.Linear(256 + 3, 32)
        self.xy_f2 = nn.Linear(32 + 3, 2)
        self.initialize_weights()

    def forward(self, rgb, rendered, pred_rot, pred_trans):
        feat_rgb = self.first_block_rgb(rgb)
        feat_rendered = self.first_block_rendered(rendered)

        middle = self.second_block(feat_rgb - feat_rendered).view(-1, 512)
        middle = torch.relu(self.top_fc(middle))

        # Not sure if this is what they thought, but I can not
        # came up with any other reasonable implementation.
        rotation = torch.relu(self.rot_f1(torch.cat([middle, pred_rot], dim=1)))
        rotation = self.rot_f2(torch.cat([rotation, pred_rot], dim=1))

        # Suprisingly z prediction head from paper did not use predicted z.
        # Not sure why. Decided to use it anyway to enable identity initalization.
        z = torch.relu(self.z_f1(torch.cat([middle, pred_trans[:, 2].view(-1, 1)], dim=1)))
        z = self.z_f2(torch.cat([z, pred_trans[:, 2].view(-1, 1)], dim=1)).view(-1, 1)

        xy = torch.relu(self.xy_f1(torch.cat([middle, pred_trans[:, :2], z], dim=1)))
        xy = self.xy_f2(torch.cat([xy, pred_trans[:, :2], z], dim=1)).view(-1, 2)

        return rotation, torch.cat([xy, z], dim=1)

    def initialize_weights(self):
        for l in [self.rot_f1, self.rot_f2, self.z_f1, self.z_f2, self.xy_f1, self.xy_f2]:
            nn.init.zeros_(l.bias)

        for l in [self.rot_f1, self.z_f1, self.xy_f1]:
            nn.init.zeros_(l.weight)

        dummy = torch.zeros_like(self.rot_f2.weight)
        torch.nn.init.eye_(dummy[:, 32:])
        self.rot_f2.weight.data = dummy

        dummy = torch.zeros_like(self.z_f2.weight)
        dummy[-1] = 1
        self.z_f2.weight.data = dummy

        dummy = torch.zeros_like(self.xy_f2.weight)
        torch.nn.init.eye_(dummy[:, 32:-1])
        self.xy_f2.weight.data = dummy

