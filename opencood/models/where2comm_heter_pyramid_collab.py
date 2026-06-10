# -*- coding: utf-8 -*-

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.fuse_modules.pyramid_fuse_where2comm import Where2commPyramidFusion


class _Where2commPyramidFusionAdapter(Where2commPyramidFusion):
    """
    Adapter that lets the standard HeterPyramidCollab forward path consume the
    Where2comm pyramid module without changing inference code.
    """

    def forward_collab(self, *args, **kwargs):
        fused_feature, communication_rates, _, occ_outputs = super().forward_collab(
            *args, **kwargs
        )
        self.last_comm_rate = communication_rates
        return fused_feature, occ_outputs


class Where2commHeterPyramidCollab(HeterPyramidCollab):
    """
    Heterogeneous pyramid detector using Where2comm for region selection.
    """

    def __init__(self, args):
        super().__init__(args)
        self.pyramid_backbone = _Where2commPyramidFusionAdapter(args["fusion_backbone"])
