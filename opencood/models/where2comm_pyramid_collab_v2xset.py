from opencood.models.blindmap_pyramid_collab_v2xset import BlindmapPyramidCollabV2xset
from opencood.models.fuse_modules.pyramid_fuse_where2comm import Where2commPyramidFusion


class Where2commPyramidCollabV2xset(BlindmapPyramidCollabV2xset):
    """
    OPV2V/V2XSet LiDAR pyramid model with the original Where2comm communication module.
    """

    def __init__(self, args):
        super().__init__(args)
        self.pyramid_backbone = Where2commPyramidFusion(args['fusion_backbone'])

    def forward(self, data_dict):
        return self.forward_colla(data_dict)

    def forward_colla(self, data_dict):
        output_dict = super().forward_colla(data_dict)
        output_dict.pop('pred_blind_maps', None)
        return output_dict
