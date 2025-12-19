from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import models

__paper__ = "https://arxiv.org/pdf/1611.06612.pdf"
__reference__ = "https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/refine_net.py"


def convolution_3x3(in_planes, out_planes, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=padding,
        bias=bias,
    )


def convolution_1x1(in_planes, out_planes, stride=1, padding=0, bias=True):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 1),
        stride=(stride, stride),
        padding=padding,
        bias=bias,
    )


def _create_4channel_conv1(pretrained_conv1_3ch: nn.Conv2d) -> nn.Conv2d:
    import torch

    out_channels = pretrained_conv1_3ch.out_channels
    kernel_size = pretrained_conv1_3ch.kernel_size
    stride = pretrained_conv1_3ch.stride
    padding = pretrained_conv1_3ch.padding
    bias = pretrained_conv1_3ch.bias is not None

    conv1_4ch = nn.Conv2d(
        in_channels=4,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    with torch.no_grad():
        conv1_4ch.weight[:, 0:3, :, :] = pretrained_conv1_3ch.weight.clone()
        conv1_4ch.weight[:, 3:4, :, :] = pretrained_conv1_3ch.weight.mean(dim=1, keepdim=True)

        if bias and pretrained_conv1_3ch.bias is not None:
            conv1_4ch.bias = pretrained_conv1_3ch.bias.clone()

    return conv1_4ch


class ResidualConvolutionUnit(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.convolution_layer_1 = convolution_3x3(in_planes, out_planes)
        self.convolution_layer_2 = convolution_3x3(out_planes, out_planes)

    def forward(self, x):
        residual = x
        x = self.non_linearity(x)
        x = self.convolution_layer_1(x)
        x = self.non_linearity(x)
        x = self.convolution_layer_2(x)
        x = residual + x
        return x


class MultiResolutionFusion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.convolution_layer_lower_inputs = convolution_3x3(in_planes, out_planes)
        self.convolution_layer_higher_inputs = convolution_3x3(out_planes, out_planes)

    def forward(self, backbone_features, refine_block_features=None):
        if refine_block_features is None:
            return self.convolution_layer_higher_inputs(backbone_features)
        else:
            backbone_features = self.convolution_layer_higher_inputs(backbone_features)
            refine_block_features = self.convolution_layer_lower_inputs(refine_block_features)
            refine_block_features = F.interpolate(
                refine_block_features,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            return refine_block_features + backbone_features


class ChainedResidualPooling(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.convolution_layer_1 = convolution_3x3(in_planes=in_planes, out_planes=out_planes)
        self.max_pooling_layer = nn.MaxPool2d((5, 5), stride=1, padding=2)

    def forward(self, x):
        x_non_linearity = self.non_linearity(x)
        first_pass = self.max_pooling_layer(x_non_linearity)
        first_pass = self.convolution_layer_1(first_pass)
        intermediate_sum = first_pass + x_non_linearity
        second_pass = self.max_pooling_layer(first_pass)
        second_pass = self.convolution_layer_1(second_pass)
        x = second_pass + intermediate_sum
        return x


class RefineBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.residual_convolution_unit = ResidualConvolutionUnit(out_planes, out_planes)
        self.multi_resolution_fusion = MultiResolutionFusion(in_planes, out_planes)
        self.chained_residual_pooling = ChainedResidualPooling(out_planes, out_planes)

    def forward(self, backbone_features, refine_block_features=None):
        if refine_block_features is None:
            x = self.residual_convolution_unit(backbone_features)
            x = self.residual_convolution_unit(x)
            x = self.multi_resolution_fusion(x)
            x = self.chained_residual_pooling(x)
            x = self.residual_convolution_unit(x)
            return x
        else:
            x = self.residual_convolution_unit(backbone_features)
            x = self.residual_convolution_unit(x)
            x = self.multi_resolution_fusion(x, refine_block_features)
            x = self.chained_residual_pooling(x)
            x = self.residual_convolution_unit(x)
            return x


class ReFineNet4Ch(nn.Module):
    """
    ReFineNet with 4-channel input support (RGB + OSM mask).
    
    Args:
        input_channels: 3 (RGB only) or 4 (RGB + OSM mask)
        res_net_to_use: "resnet50" or "resnet34"
        pre_trained_image_net: Whether to use ImageNet pretrained weights
        top_layers_trainable: Whether backbone layers are trainable
        num_classes: Number of output classes (1 for binary segmentation)
    """

    def __init__(
        self,
        input_channels: int = 4,
        res_net_to_use: str = "resnet50",
        pre_trained_image_net: bool = True,
        top_layers_trainable: bool = True,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        if input_channels not in (3, 4):
            raise ValueError(f"input_channels must be 3 or 4, got {input_channels}")

        res_net = getattr(models, res_net_to_use)(pretrained=pre_trained_image_net)

        if not top_layers_trainable:
            for param in res_net.parameters():
                param.requires_grad = False

        if input_channels == 4:
            import torch
            conv1_4ch = _create_4channel_conv1(res_net.conv1)
            # Also update batch norm to match (though it's channel-agnostic, we keep it for consistency)
            bn1 = res_net.bn1
            self.layer0 = nn.Sequential(conv1_4ch, bn1, res_net.relu, res_net.maxpool)
        else:
            self.layer0 = nn.Sequential(res_net.conv1, res_net.bn1, res_net.relu, res_net.maxpool)

        self.layer1 = res_net.layer1
        self.layer2 = res_net.layer2
        self.layer3 = res_net.layer3
        self.layer4 = res_net.layer4

        if res_net_to_use == "resnet50":
            layers_features = [256, 512, 1024, 2048]
        elif res_net_to_use == "resnet34":
            layers_features = [64, 128, 256, 512]
        else:
            raise NotImplementedError

        self.convolution_layer_4_dim_reduction = convolution_3x3(
            in_planes=layers_features[-1], out_planes=512
        )
        self.convolution_layer_3_dim_reduction = convolution_3x3(
            in_planes=layers_features[-2], out_planes=256
        )
        self.convolution_layer_2_dim_reduction = convolution_3x3(
            in_planes=layers_features[-3], out_planes=256
        )
        self.convolution_layer_1_dim_reduction = convolution_3x3(
            in_planes=layers_features[-4], out_planes=256
        )

        self.refine_block_4 = RefineBlock(in_planes=512, out_planes=512)
        self.refine_block_3 = RefineBlock(in_planes=512, out_planes=256)
        self.refine_block_2 = RefineBlock(in_planes=256, out_planes=256)
        self.refine_block_1 = RefineBlock(in_planes=256, out_planes=256)

        self.residual_convolution_unit = ResidualConvolutionUnit(in_planes=256, out_planes=256)
        self.final_layer = convolution_1x1(in_planes=256, out_planes=self.num_classes)

    def forward(self, input_feature: Tensor) -> Tensor:
        layer_0_output = self.layer0(input_feature)
        layer_1_output = self.layer1(layer_0_output)
        layer_2_output = self.layer2(layer_1_output)
        layer_3_output = self.layer3(layer_2_output)
        layer_4_output = self.layer4(layer_3_output)

        backbone_layer_4 = self.convolution_layer_4_dim_reduction(layer_4_output)
        backbone_layer_3 = self.convolution_layer_3_dim_reduction(layer_3_output)
        backbone_layer_2 = self.convolution_layer_2_dim_reduction(layer_2_output)
        backbone_layer_1 = self.convolution_layer_1_dim_reduction(layer_1_output)

        refine_block_4 = self.refine_block_4(backbone_layer_4)
        refine_block_3 = self.refine_block_3(backbone_layer_3, refine_block_4)
        refine_block_3 = self.refine_block_2(backbone_layer_2, refine_block_3)
        refine_block_1 = self.refine_block_1(backbone_layer_1, refine_block_3)

        residual_convolution_unit = self.residual_convolution_unit(refine_block_1)
        residual_convolution_unit = self.residual_convolution_unit(residual_convolution_unit)

        final_map = self.final_layer(residual_convolution_unit)
        final_map = F.interpolate(final_map, scale_factor=4, mode="bilinear", align_corners=True)
        return final_map

