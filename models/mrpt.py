"""
mrpy.py 2022/12/10 20:04
Written by Wensheng Fan
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.basic_block import BasicBlock
from models.modules.bottleneck_block import Bottleneck, BottleneckDWP
from models.modules.transformer_block import GeneralTransformerBlock
from models.modules.rcab import CALayer
from models.modules.resblock import ResBlock
from utils import feature_visualization, feat_split_visualize
from pathlib import Path

blocks_dict = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
    "TRANSFORMER_BLOCK": GeneralTransformerBlock,
}

BN_MOMENTUM = 0.1


class HighResolutionModule(nn.Module):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            num_input_resolutions,
            attn_types,
            ffn_types,
            multi_scale_output=True,
            out_scales=0,
            drop_paths=0.0,
    ):
        """
        Normal Args:
            :param num_branches:
            :param blocks:
            :param num_blocks:
            :param num_inchannels:
            :param num_channels:
            :param multi_scale_output:
            :param out_scales:
        Transformer_related Args:
            :param num_heads: the number of head witin each MHSA
            :param num_window_sizes: the window size for the local self-attention
            :param num_mlp_ratios:
            :param num_input_resolutions: the spatial height/width of the input feature maps.
            :param attn_types:
            :param ffn_types:
            :param drop_paths:
        """
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.out_scales = out_scales

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.attn_types = attn_types
        self.ffn_types = ffn_types

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
            self,
            branch_index,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
            stride=1
    ):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index]
                != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            ) if block == BasicBlock or block == Bottleneck else block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                input_resolution=num_input_resolutions[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                attn_type=attn_types[branch_index][0],
                ffn_type=ffn_types[branch_index][0],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                ) if block == BasicBlock or block == Bottleneck else block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    input_resolution=num_input_resolutions[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    attn_type=attn_types[branch_index][i],
                    ffn_type=ffn_types[branch_index][i],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(self,
                       num_branches,
                       block,
                       num_blocks,
                       num_channels,
                       num_input_resolutions,
                       num_heads,
                       num_window_sizes,
                       num_mlp_ratios,
                       attn_types,
                       ffn_types,
                       drop_paths,
                       ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else self.out_scales):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                            ),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        num_inchannels = self.num_inchannels
        if not self.multi_scale_output and self.out_scales:
            num_inchannels = self.num_inchannels[0: self.out_scales]
        return num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRModuleWOFusion(nn.Module):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            num_input_resolutions,
            attn_types,
            ffn_types,
            multi_scale_output=True,
            out_scales=0,
            drop_paths=0.0,
    ):
        super(HRModuleWOFusion, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.out_scales = out_scales

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
        )
        self.relu = nn.ReLU()

        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.attn_types = attn_types
        self.ffn_types = ffn_types

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         num_input_resolutions,
                         num_heads,
                         num_window_sizes,
                         num_mlp_ratios,
                         attn_types,
                         ffn_types,
                         drop_paths,
                         stride=1
                         ):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index]
                != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            ) if block == BasicBlock or block == Bottleneck else block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                input_resolution=num_input_resolutions[branch_index],
                num_heads=num_heads[branch_index],
                window_size=num_window_sizes[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                attn_type=attn_types[branch_index][0],
                ffn_type=ffn_types[branch_index][0],
                drop_path=drop_paths[0],
            )
        )

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                ) if block == BasicBlock or block == Bottleneck else block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    input_resolution=num_input_resolutions[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    attn_type=attn_types[branch_index][i],
                    ffn_type=ffn_types[branch_index][i],
                    drop_path=drop_paths[i],
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(
            self,
            num_branches,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def get_num_inchannels(self):
        num_inchannels = self.num_inchannels
        if not self.multi_scale_output and self.out_scales:
            num_inchannels = self.num_inchannels[0: self.out_scales]
        return num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        out = []
        for i in range(self.num_branches):
            y = x[i]
            out.append(self.relu(y))

        return out


class RCAFB(nn.Module):
    def __init__(
            self, in_chans, out_chans, reduction=16, expansion=4, m_chans=0):

        super(RCAFB, self).__init__()
        if m_chans == 0:
            m_chans = in_chans * expansion
        self.expan_conv = nn.Sequential(
            nn.Conv2d(in_chans, m_chans, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            CALayer(m_chans, reduction),
            nn.Conv2d(m_chans, m_chans, kernel_size=3, stride=1, padding=1),
        )
        self.compression = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(m_chans, out_chans, kernel_size=1)
        )

    def forward(self, x):
        x = self.expan_conv(x)
        res = x
        res = self.body(res)
        res += x
        res = self.compression(res)
        return res


class MRPT(nn.Module):
    def __init__(self, cfg, ms_chans=4, img_size=256, config=None):
        super(MRPT, self).__init__()
        self.ms_chans = ms_chans
        self.img_size = img_size
        self.config = config
        shallow_c = cfg["STAGE1"]["NUM_CHANNELS"]

        self.hrstem_m = nn.Sequential(
            nn.Conv2d(ms_chans, shallow_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(shallow_c, shallow_c, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
        )
        self.hrstem_p = nn.Sequential(
            nn.Conv2d(1, shallow_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(shallow_c, shallow_c, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
        )

        # stochastic depth
        depth_s2_m = max(cfg["MS_STAGE2"]["NUM_BLOCKS"]) * cfg["MS_STAGE2"]["NUM_MODULES"]
        depth_s3_m = max(cfg["MS_STAGE3"]["NUM_BLOCKS"]) * cfg["MS_STAGE3"]["NUM_MODULES"]
        depths_m = [depth_s2_m, depth_s3_m]
        drop_path_rate = cfg["DROP_PATH_RATE"]
        dpr_m = [x.item() for x in torch.linspace(0, drop_path_rate / 2, sum(depths_m))]

        depth_s2_p = max(cfg["PAN_STAGE2"]["NUM_BLOCKS"]) * cfg["PAN_STAGE2"]["NUM_MODULES"]
        depth_s3_p = max(cfg["PAN_STAGE3"]["NUM_BLOCKS"]) * cfg["PAN_STAGE3"]["NUM_MODULES"]
        depths_p = [depth_s2_p, depth_s3_p]
        dpr_p = [x.item() for x in torch.linspace(0, drop_path_rate / 2, sum(depths_p))]

        #self.downsample_ms = nn.Sequential(
        #    nn.Conv2d(shallow_c, shallow_c, kernel_size=3, stride=2, padding=1),
        #    nn.ReLU(inplace=True)
        #)
        #self.downsample_pan = nn.Sequential(
        #    nn.Conv2d(shallow_c, shallow_c, kernel_size=3, stride=2, padding=1),
        #    nn.ReLU(inplace=True)
        #)

        self.ms_stage2_cfg = cfg["MS_STAGE2"]
        num_channels = self.ms_stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.ms_stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.ms_transition1 = self._make_transition_layer(
            [shallow_c], num_channels
        )
        self.ms_stage2, pre_stage_channels = self._make_stage(
            self.ms_stage2_cfg, num_channels, drop_paths=dpr_m[0:depth_s2_m]
        )

        self.ms_stage3_cfg = cfg["MS_STAGE3"]
        num_channels = self.ms_stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.ms_stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.ms_transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.ms_stage3, ms_final_channels = self._make_stage(
            self.ms_stage3_cfg, num_channels, multi_scale_output=True, drop_paths=dpr_m[depth_s2_m:]
        )

        self.pan_stage2_cfg = cfg["PAN_STAGE2"]
        num_channels = self.pan_stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.pan_stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.pan_transition1 = self._make_transition_layer(
            [shallow_c], num_channels
        )
        self.pan_stage2, pre_stage_channels = self._make_stage(
            self.pan_stage2_cfg, num_channels, drop_paths=dpr_p[0:depth_s2_p]
        )

        self.pan_stage3_cfg = cfg["PAN_STAGE3"]
        num_channels = self.pan_stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.pan_stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.pan_transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.pan_stage3, pan_final_channels = self._make_stage(
            self.pan_stage3_cfg, num_channels, multi_scale_output=True, drop_paths=dpr_p[depth_s2_p:]
        )

        pre_stage_channels = [
            ms_final_channels[i] + pan_final_channels[i] for i in range(len(ms_final_channels))
        ]
        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        self.fuse1 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[0], num_channels[0], kernel_size=3, stride=1, padding=1),
            ResBlock(num_channels[0], num_channels[0]),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[1], num_channels[1], kernel_size=3, stride=1, padding=1),
            ResBlock(num_channels[1], num_channels[1]),
            nn.Conv2d(num_channels[1], num_channels[1] * 4, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(num_channels[1], num_channels[1], 1, 1, 0,),
            #nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[2], num_channels[2], kernel_size=3, stride=1, padding=1),
            ResBlock(num_channels[2], num_channels[2]),
            nn.Conv2d(num_channels[2], num_channels[2] * 4, kernel_size=3, stride=1, padding=1),
            #nn.Conv2d(num_channels[1], num_channels[1], 1, 1, 0,),
            #nn.Upsample(scale_factor=4, mode="nearest"),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(num_channels[2], num_channels[2] * 4, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(num_channels[1], num_channels[1], 1, 1, 0,),
            #nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

        pre_stage_channels = sum(num_channels, shallow_c*2)

        self.recon_head = RCAFB(pre_stage_channels, ms_chans, reduction=4, expansion=1)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, out_scales=0, drop_paths=0.0):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        num_heads = layer_config["NUM_HEADS"]
        num_window_sizes = layer_config["NUM_WINDOW_SIZES"]
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]
        num_input_resolutions = layer_config["NUM_RESOLUTIONS"]
        attn_types = layer_config["ATTN_TYPES"]
        ffn_types = layer_config["FFN_TYPES"]

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    num_input_resolutions,
                    attn_types[i],
                    ffn_types[i],
                    reset_multi_scale_output,
                    out_scales,
                    drop_paths=drop_paths[max(num_blocks) * i: max(num_blocks) * (i + 1)],
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        input_pan, input_ms = x[:, self.ms_chans, :, :].unsqueeze(1), x[:, :self.ms_chans, :, :]
        ms_feat = self.hrstem_m(input_ms)
        pan_feat = self.hrstem_p(input_pan)

        #ms_down = self.downsample_ms(ms_feat)
        #pan_down = self.downsample_ms(pan_feat)

        scale_added_list = []
        for i in range(self.ms_stage2_cfg["NUM_BRANCHES"]):
            if self.ms_transition1[i] is not None:
                scale_added_list.append(self.ms_transition1[i](ms_feat))
            else:
                scale_added_list.append(ms_feat)
        ms_list = self.ms_stage2(scale_added_list)

        scale_added_list = []
        for i in range(self.pan_stage2_cfg["NUM_BRANCHES"]):
            if self.pan_transition1[i] is not None:
                scale_added_list.append(self.pan_transition1[i](pan_feat))
            else:
                scale_added_list.append(pan_feat)
        pan_list = self.pan_stage2(scale_added_list)

        scale_added_list = []
        for i in range(self.ms_stage3_cfg["NUM_BRANCHES"]):
            if self.ms_transition2[i] is not None:
                scale_added_list.append(self.ms_transition2[i](ms_list[i if i < len(ms_list) else -1]))
            else:
                scale_added_list.append(ms_list[i])
        ms_list = self.ms_stage3(scale_added_list)

        scale_added_list = []
        for i in range(self.pan_stage3_cfg["NUM_BRANCHES"]):
            if self.pan_transition2[i] is not None:
                scale_added_list.append(self.pan_transition2[i](pan_list[i if i < len(pan_list) else -1]))
            else:
                scale_added_list.append(pan_list[i])
        pan_list = self.pan_stage3(scale_added_list)

        fused1 = self.fuse1(torch.cat((ms_list[0], pan_list[0]), dim=1))
        fused2 = self.fuse2(torch.cat((ms_list[1], pan_list[1]), dim=1))
        fused3 = self.fuse3(torch.cat((ms_list[2], pan_list[2]), dim=1))

        fused_feat = torch.cat((ms_feat, pan_feat, fused1, fused2, fused3), dim=1)

        # final_features = self.fuse_stage3(fused_feat)

        out = self.recon_head(fused_feat) + input_ms

        # visualize fus_feat, ms_feat3, ms_feat2, ms_feat1, pan_feat1
        feat_split_visualize(ms_feat, module_type='preconv_m', stage='', save_dir=Path(self.config['OUTPUT']))
        feat_split_visualize(pan_feat, module_type='preconv_p', stage='', save_dir=Path(self.config['OUTPUT']))
        feat_split_visualize(fused1, module_type='mrfm1', stage='', save_dir=Path(self.config['OUTPUT']))
        feat_split_visualize(fused2, module_type='mrfm2', stage='', save_dir=Path(self.config['OUTPUT']))
        feat_split_visualize(fused3, module_type='mrfm3', stage='', save_dir=Path(self.config['OUTPUT']))

        return out

if __name__ == '__main__':
    import argparse
    from cfg import get_config
    from torchsummary import summary


    def parse_option():
        parser = argparse.ArgumentParser('HRPFormer training and evaluation script', add_help=False)
        parser.add_argument(
            "--cfg",
            type=str,
            metavar="FILE",
            default=r"../configs/mrpt/mrpt_x.yaml",
            help="path to config file, model name and type are in the yaml cfg file",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs="+",
        )

        # easy config modification
        parser.add_argument("--batch-size", type=int, default=1, help="batch size for single GPU")
        parser.add_argument("--tag", type=str, default="qb", help="tag of experiment")
        parser.add_argument("--ckpt-choice", type=str, default="latest",
                            choices=["best_ergas", "min_loss", "latest", "best_sam", "best_qnr", "80"],
                            help="the model to be tested")
        parser.add_argument("--test", action="store_true", help="Perform test only")
        parser.add_argument(
            "--throughput", action="store_true", help="Test throughput only"
        )
        # set training dataset
        parser.add_argument(
            "--data-set", default="qb", choices=["gf2", "qb", "wv3", "gf2l", "spot", "qbsx"], type=str
        )
        parser.add_argument(
            "--tsp", default="", type=str, help="Training Set Path"
        )
        parser.add_argument('--base-lr', type=float, default=0.0005, help='basic learning rate')
        parser.add_argument('--num-workers', type=int, default=1,
                            help='threads for data loading, too many would cause error, for ur own pc suggest 1')
        parser.add_argument("--verbose", action="store_false", help="output verbose for analysis")
        parser.add_argument('--gpu-set', type=int, default=1,
                            help='if >1, should modify the code otherwise wont be valid')

        args, unparsed = parser.parse_known_args()
        config = get_config(args)
        return args, config


    _, config = parse_option()
    #N = MRPT(cfg=config.MODEL.MRPT, ms_chans=4, img_size=128).cuda()
    #N = MRPTWOMRFEF(cfg=config.MODEL.MRPT, ms_chans=4, img_size=128).cuda()
    #N = MRPTWOMRFM(cfg=config.MODEL.MRPT, ms_chans=4, img_size=128).cuda()
    #N = MRPTWOSDFM(cfg=config.MODEL.MRPT, ms_chans=4, img_size=128).cuda()
    N = MRPTWOSC(cfg=config.MODEL.MRPT, ms_chans=4, img_size=128).cuda()

    # 显示网络中每一层的参数量，简单好用
    def model_structure(model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  ##如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

    model_structure(N)
    #summary(N, [(5, 128, 128)], device='cuda')