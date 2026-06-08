from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CommSimulatorConfig:
    packet_size: int = 8
    packet_loss_prob: float = 0.0
    collaborator_dropout_prob: float = 0.0
    base_latency_ms: float = 0.0
    jitter_std_ms: float = 0.0
    queue_delay_mean_ms: float = 0.0
    deadline_ms: Optional[float] = None
    max_retransmissions: int = 0
    retransmission_delay_ms: float = 5.0
    loss_model: str = "iid"
    ge_good_to_bad: float = 0.02
    ge_bad_to_good: float = 0.20
    ge_good_loss: float = 0.01
    ge_bad_loss: float = 0.50
    keep_ego: bool = True
    seed: Optional[int] = None


class CommSimulator:
    """
    Packet-level communication perturbation for selected BEV feature regions.

    The simulator operates on communication masks after region selection and
    before fusion. It does not change the original selection policy.
    """

    def __init__(self, config: CommSimulatorConfig):
        self.config = config
        self.generators = {}

    @classmethod
    def from_dict(cls, cfg: Dict):
        return cls(CommSimulatorConfig(**cfg))

    def __call__(
        self,
        communication_masks: torch.Tensor,
        record_len,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        device = communication_masks.device
        dtype = communication_masks.dtype
        total_agents, _, height, width = communication_masks.shape
        packet_h = max(1, int(cfg.packet_size))
        packet_w = packet_h
        grid_h = (height + packet_h - 1) // packet_h
        grid_w = (width + packet_w - 1) // packet_w

        pooled = F.max_pool2d(
            communication_masks.float(),
            kernel_size=(packet_h, packet_w),
            stride=(packet_h, packet_w),
            ceil_mode=True,
        )
        packet_selected = pooled > 0
        packet_delivered = packet_selected.clone()

        start = 0
        dropped_collaborators = 0
        for num_cav in record_len:
            num_cav_int = int(num_cav)
            for local_idx in range(num_cav_int):
                global_idx = start + local_idx
                if cfg.keep_ego and local_idx == 0:
                    continue
                if self._rand((), device).item() < cfg.collaborator_dropout_prob:
                    packet_delivered[global_idx] = False
                    dropped_collaborators += 1
            start += num_cav_int

        loss_mask = self._sample_packet_loss(packet_delivered.shape, device)
        if cfg.max_retransmissions > 0:
            loss_mask = self._apply_retransmissions(loss_mask, device)
        packet_delivered = packet_delivered & (~loss_mask)

        latency_ms = self._sample_latency(packet_delivered.shape, device)
        if cfg.deadline_ms is not None:
            packet_delivered = packet_delivered & (latency_ms <= cfg.deadline_ms)

        if cfg.keep_ego:
            start = 0
            for num_cav in record_len:
                packet_delivered[start] = packet_selected[start]
                start += int(num_cav)

        delivered_mask = F.interpolate(
            packet_delivered.to(dtype),
            size=(height, width),
            mode="nearest",
        )
        delivered_mask = delivered_mask[:, :, :height, :width]
        communication_masks = communication_masks * delivered_mask

        transmitted_packet_mask = torch.ones_like(packet_selected, dtype=torch.bool)
        start = 0
        for num_cav in record_len:
            transmitted_packet_mask[start] = False
            start += int(num_cav)

        transmitted_selected = packet_selected & transmitted_packet_mask
        transmitted_delivered = transmitted_selected & packet_delivered
        selected_packets = transmitted_selected.sum().item()
        delivered_packets = transmitted_delivered.sum().item()
        total_transmitted_packets = transmitted_packet_mask.sum().item()
        deadline_drops = (
            (transmitted_selected & (latency_ms > cfg.deadline_ms)).sum().item()
            if cfg.deadline_ms is not None
            else 0
        )
        stats = {
            "selected_rate": float(selected_packets / max(total_transmitted_packets, 1)),
            "delivered_rate": float(delivered_packets / max(total_transmitted_packets, 1)),
            "effective_mask_rate": self._mask_rate(communication_masks),
            "selected_packets": float(selected_packets),
            "delivered_packets": float(delivered_packets),
            "packet_delivery_ratio": float(delivered_packets / max(selected_packets, 1)),
            "dropped_collaborators": float(dropped_collaborators),
            "deadline_dropped_packets": float(deadline_drops),
            "mean_latency_ms": float(latency_ms[transmitted_selected].mean().item())
            if selected_packets > 0
            else 0.0,
        }
        return communication_masks, stats

    @staticmethod
    def _mask_rate(mask: torch.Tensor) -> float:
        if mask.numel() == 0:
            return 0.0
        return float((mask > 0).float().mean().item())

    def _rand(self, shape, device):
        return torch.rand(shape, generator=self._get_generator(device), device=device)

    def _randn(self, shape, device):
        return torch.randn(shape, generator=self._get_generator(device), device=device)

    def _get_generator(self, device):
        if self.config.seed is None:
            return None
        key = str(device)
        if key not in self.generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.seed)
            self.generators[key] = generator
        return self.generators[key]

    def _sample_packet_loss(self, shape, device):
        cfg = self.config
        if cfg.loss_model == "iid":
            return self._rand(shape, device) < cfg.packet_loss_prob
        if cfg.loss_model != "gilbert_elliott":
            raise ValueError(f"Unsupported loss_model: {cfg.loss_model}")

        flat_size = 1
        for dim in shape:
            flat_size *= dim
        state_bad = torch.zeros(flat_size, dtype=torch.bool, device=device)
        losses = torch.zeros(flat_size, dtype=torch.bool, device=device)
        for idx in range(flat_size):
            if state_bad[idx]:
                state_bad[idx] = self._rand((), device) >= cfg.ge_bad_to_good
            else:
                state_bad[idx] = self._rand((), device) < cfg.ge_good_to_bad
            loss_prob = cfg.ge_bad_loss if state_bad[idx] else cfg.ge_good_loss
            losses[idx] = self._rand((), device) < loss_prob
        return losses.view(shape)

    def _apply_retransmissions(self, initial_loss, device):
        cfg = self.config
        remaining_loss = initial_loss.clone()
        for _ in range(cfg.max_retransmissions):
            retry_loss = self._sample_packet_loss(initial_loss.shape, device)
            remaining_loss = remaining_loss & retry_loss
        return remaining_loss

    def _sample_latency(self, shape, device):
        cfg = self.config
        latency = torch.full(shape, float(cfg.base_latency_ms), device=device)
        if cfg.jitter_std_ms > 0:
            jitter = self._randn(shape, device) * cfg.jitter_std_ms
            latency = latency + torch.clamp(jitter, min=0.0)
        if cfg.queue_delay_mean_ms > 0:
            uniform = torch.clamp(self._rand(shape, device), min=1e-6)
            latency = latency - torch.log(uniform) * cfg.queue_delay_mean_ms
        if cfg.max_retransmissions > 0:
            latency = latency + cfg.max_retransmissions * cfg.retransmission_delay_ms
        return latency
