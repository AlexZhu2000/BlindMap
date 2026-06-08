# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.comm_modules.where2comm import Communication
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import Bottleneck, ResNetModified
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class MaxFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]


class SumFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, dim=0)


class Where2commPyramidFusion(ResNetBEVBackbone):
    """
    Pyramid fusion with the original Where2comm communication module.

    This keeps the BlindMap pyramid backbone/fusion layout unchanged and only
    replaces the region-selection module with opencood.models.comm_modules.where2comm.Communication.
    """

    def __init__(self, model_cfg, input_channels=64):
        print('Where2commPyramidFusion')
        super().__init__(model_cfg, input_channels)

        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(
                Bottleneck,
                self.model_cfg['layer_nums'],
                self.model_cfg['layer_strides'],
                self.model_cfg['num_filters'],
                inplanes=model_cfg.get('inplanes', 64),
                groups=32,
                width_per_group=4,
            )

        self.where2comm_communication = Communication(self.model_cfg["communication"])
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        self.agg_mode = self.model_cfg["communication"].get('fusion_mode', "MAX")

        self.fuse_modules = nn.ModuleList()
        for _ in range(self.num_levels):
            if self.agg_mode == "MAX":
                fuse_network = MaxFusion()
            elif self.agg_mode == "SUM":
                fuse_network = SumFusion()
            else:
                raise ValueError(f"Unsupported fusion_mode: {self.agg_mode}")
            self.fuse_modules.append(fuse_network)

        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map_list.append(eval(f"self.single_head_{i}")(feature_list[i]))
        final_feature = self.decode_multiscale_feature(feature_list)
        return final_feature, occ_map_list

    def forward_collab(
        self,
        spatial_features,
        record_len,
        affine_matrix,
        agent_modality_list=None,
        cam_crop_info=None,
        history_blind_maps=None,
    ):
        B = affine_matrix.shape[0]

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []

        for i in range(self.num_levels):
            x = feature_list[i]
            _, _, curr_H, curr_W = x.shape

            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)

            if i == 0:
                batch_confidence_maps = regroup(occ_map, record_len)
                (
                    batch_communication_maps_list,
                    communication_masks,
                    communication_rates,
                ) = self.where2comm_communication(
                    batch_confidence_maps,
                    record_len,
                    affine_matrix,
                )
                original_maps = torch.cat(batch_communication_maps_list, dim=0)
                x = x * communication_masks
            else:
                resized_maps = F.interpolate(
                    original_maps,
                    size=(curr_H, curr_W),
                    mode='bilinear',
                    align_corners=True,
                )
                resized_mask = F.interpolate(
                    communication_masks,
                    size=(curr_H, curr_W),
                    mode='bilinear',
                    align_corners=True,
                )
                x = x * resized_mask

            batch_node_features = regroup(x, record_len)
            batch_communication_maps = regroup(
                resized_maps if i > 0 else original_maps,
                record_len,
            )

            x_fuse = []
            for b in range(B):
                N = record_len[b]
                t_matrix = affine_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]
                _, H, W = node_features.shape[1:]
                neighbor_feature = warp_affine_simple(
                    node_features,
                    t_matrix[0, :, :, :],
                    (H, W),
                )

                if self.agg_mode == "Blindfusion":
                    curr_comm_maps = batch_communication_maps[b]
                    neighbor_comm_maps = warp_affine_simple(
                        curr_comm_maps,
                        t_matrix[0, :, :, :],
                        (H, W),
                        mode='bilinear',
                    )
                    x_fuse.append(self.fuse_modules[i](neighbor_feature, neighbor_comm_maps))
                else:
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))

            fused_feature_list.append(torch.stack(x_fuse))

        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        return fused_feature, communication_rates, None, occ_map_list
