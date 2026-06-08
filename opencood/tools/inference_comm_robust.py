import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict
from opencood.models.comm_modules.blindcomm_robust import BlindCommunicationRobust


def parser():
    parser = argparse.ArgumentParser(description="communication robust inference")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--fusion_method", type=str, default="intermediate")
    parser.add_argument("--range", type=str, default="102.4,102.4")
    parser.add_argument("--modal", type=int, default=0)
    parser.add_argument("--noise", default="0,0,0,0", type=str)
    parser.add_argument("--comm_volume_MB", type=float, default=None)
    parser.add_argument("--comm_thre", type=float, default=None)
    parser.add_argument("--time_delay", type=int, default=0)
    parser.add_argument("--bandwidth", type=int, default=None)
    parser.add_argument("--packet_loss_prob", type=float, default=0.0)
    parser.add_argument("--collab_dropout_prob", type=float, default=0.0)
    parser.add_argument("--queue_delay_mean_ms", type=float, default=0.0)
    parser.add_argument("--jitter_std_ms", type=float, default=0.0)
    parser.add_argument("--deadline_ms", type=float, default=None)
    parser.add_argument("--max_retransmissions", type=int, default=0)
    parser.add_argument("--packet_size", type=int, default=8)
    parser.add_argument("--loss_model", type=str, default="iid")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def build_simulator_cfg(opt):
    return {
        "packet_size": opt.packet_size,
        "packet_loss_prob": opt.packet_loss_prob,
        "collaborator_dropout_prob": opt.collab_dropout_prob,
        "queue_delay_mean_ms": opt.queue_delay_mean_ms,
        "jitter_std_ms": opt.jitter_std_ms,
        "deadline_ms": opt.deadline_ms,
        "max_retransmissions": opt.max_retransmissions,
        "loss_model": opt.loss_model,
        "seed": opt.seed,
    }


def main():
    opt = parser()
    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_backbone"]["communication"]["thre"] = opt.comm_thre
        hypes["model"]["args"]["fusion_backbone"]["communication"]["use_threshold"] = True
    if opt.comm_volume_MB is not None:
        hypes["model"]["args"]["fusion_backbone"]["communication"]["comm_volume_MB"] = opt.comm_volume_MB
        hypes["model"]["args"]["fusion_backbone"]["communication"]["use_threshold"] = False

    x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
    y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
    new_cav_range = [
        x_min,
        y_min,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
        x_max,
        y_max,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
    ]
    hypes = update_dict(
        hypes,
        {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        },
    )

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator_cfg = build_simulator_cfg(opt)
    print("Loading Model from checkpoint")
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")

    model.force_collab = True
    if hasattr(model, "pyramid_backbone") and hasattr(model.pyramid_backbone, "naive_communication"):
        base_comm = model.pyramid_backbone.naive_communication
        model.pyramid_backbone.naive_communication = BlindCommunicationRobust(
            hypes["model"]["args"]["fusion_backbone"]["communication"],
            hypes["model"]["args"]["fusion_backbone"]["num_filters"],
            simulator_cfg=simulator_cfg,
            base_comm=base_comm,
        )
    model.cuda() if torch.cuda.is_available() else None
    model.eval()

    print("Dataset Building")
    hypes["noise_setting"] = {
        "add_noise": False if opt.noise == "0,0,0,0" else True,
        "args": {
            "pos_std": float(opt.noise.split(",")[0]),
            "rot_std": float(opt.noise.split(",")[1]),
            "pos_mean": float(opt.noise.split(",")[2]),
            "rot_mean": float(opt.noise.split(",")[3]),
        },
    }
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}

    total_comm_rates = []
    total_sim_stats = []
    model_times = []
    infer_info = opt.fusion_method + "_commrobust"

    for i, batch_data in enumerate(data_loader):
        if i < opt.start_index:
            continue
        if opt.max_batches is not None and len(total_comm_rates) >= opt.max_batches:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            start = time.time()
            infer_result, comm_rates = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
            model_times.append(time.time() - start)
            total_comm_rates.append(comm_rates)
            robust_comm = getattr(model.pyramid_backbone, "naive_communication", None)
            if hasattr(robust_comm, "last_sim_stats"):
                total_sim_stats.append(robust_comm.last_sim_stats)

        pred_box_tensor = infer_result["pred_box_tensor"]
        gt_box_tensor = infer_result["gt_box_tensor"]
        pred_score = infer_result["pred_score"]
        for thr in [0.3, 0.5, 0.7]:
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr)

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
    comm_rates = float(sum(total_comm_rates) / max(len(total_comm_rates), 1))
    model_time_av = float(sum(model_times) / max(len(model_times), 1))
    mean_sim_stats = {}
    if total_sim_stats:
        keys = set().union(*(stats.keys() for stats in total_sim_stats))
        for key in sorted(keys):
            values = [stats.get(key, 0.0) for stats in total_sim_stats]
            mean_sim_stats[key] = float(sum(values) / len(values))
    out_path = os.path.join(opt.model_dir, "comm_robust_result.txt")
    with open(out_path, "a+") as f:
        f.write(
            f"Epoch: {resume_epoch} | AP @0.3: {ap30:.04f} | AP @0.5: {ap50:.04f} | AP @0.7: {ap70:.04f} | comm_rate: {comm_rates:.06f} | model_time: {model_time_av:.04f} | max_batches={opt.max_batches} | sim={simulator_cfg} | sim_stats={mean_sim_stats}\n"
        )


if __name__ == "__main__":
    main()
