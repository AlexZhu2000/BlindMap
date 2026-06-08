from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.comm_modules.blindcomm import BlindCommunication
from opencood.utils.comm_simulator import CommSimulator


class BlindCommunicationRobust(nn.Module):
    """
    Thin wrapper around BlindCommunication that optionally injects
    communication-level perturbations after region selection.
    """

    def __init__(
        self,
        args,
        channels_list,
        simulator_cfg: Optional[Dict] = None,
        base_comm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.base_comm = base_comm if base_comm is not None else BlindCommunication(args, channels_list)
        self.simulator = None
        self.last_sim_stats = {}
        if simulator_cfg is not None:
            self.simulator = CommSimulator.from_dict(simulator_cfg)

    def forward(
        self,
        batch_blind_maps_groups,
        record_len,
        pairwise_t_matrix,
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        batch_communication_maps_list, communication_masks, communication_rates = self.base_comm(
            batch_blind_maps_groups,
            record_len,
            pairwise_t_matrix,
        )
        self.last_sim_stats = {}
        if self.simulator is not None:
            communication_masks, self.last_sim_stats = self.simulator(
                communication_masks,
                record_len,
                pairwise_t_matrix,
            )
        return (
            batch_communication_maps_list,
            communication_masks,
            communication_rates,
        )
