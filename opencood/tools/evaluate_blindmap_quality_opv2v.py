# -*- coding: utf-8 -*-
import argparse
import importlib
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils.common_utils import update_dict

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BlindMap prediction quality on OPV2V."
    )
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--range", default="102.4,102.4", type=str)
    parser.add_argument("--noise", default="0,0,0,0", type=str)
    parser.add_argument("--time_delay", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--bins", default=10000, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument(
        "--thresholds",
        default="0.01,0.1,0.3,0.5",
        type=str,
        help="Comma separated thresholds for binary mask metrics.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def apply_range(hypes, opt_range):
    if "heter" not in hypes:
        return hypes

    x_extent, y_extent = [eval(v) for v in opt_range.split(",")]
    new_cav_range = [
        -x_extent,
        -y_extent,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
        x_extent,
        y_extent,
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

    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = None
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
            break
    if parser_func is None:
        raise RuntimeError(f"Cannot find yaml parser {hypes['yaml_parser']}")
    return parser_func(hypes)


def prepare_hypes(opt):
    hypes = yaml_utils.load_yaml(None, opt)
    hypes = apply_range(hypes, opt.range)
    hypes["time_delay"] = opt.time_delay
    hypes.setdefault("use_history", False)
    hypes["validate_dir"] = hypes["test_dir"]

    noise_opt = opt.noise.split(",")
    if len(noise_opt) != 4:
        raise ValueError("--noise should be pos_std,rot_std,pos_mean,rot_mean")
    pos_std, rot_std, pos_mean, rot_mean = (float(x) for x in noise_opt)
    hypes["noise_setting"] = OrderedDict(
        [
            ("add_noise", opt.noise != "0,0,0,0"),
            (
                "args",
                {
                    "pos_std": pos_std,
                    "rot_std": rot_std,
                    "pos_mean": pos_mean,
                    "rot_mean": rot_mean,
                },
            ),
        ]
    )
    if "box_align" in hypes:
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]
    return hypes


def collect_non_ego_predictions(pred_blind_maps, record_len):
    preds = []
    start = 0
    for num_cav in record_len.tolist():
        end = start + int(num_cav)
        if int(num_cav) > 1:
            preds.append(pred_blind_maps[start + 1:end])
        start = end
    if not preds:
        return None
    return torch.cat(preds, dim=0)


def binary_counts(pred, target, threshold):
    pred_bin = pred >= threshold
    target_bin = target >= 0.5
    tp = torch.logical_and(pred_bin, target_bin).sum().item()
    fp = torch.logical_and(pred_bin, ~target_bin).sum().item()
    fn = torch.logical_and(~pred_bin, target_bin).sum().item()
    tn = torch.logical_and(~pred_bin, ~target_bin).sum().item()
    return tp, fp, fn, tn


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def hist_curves(pos_hist, neg_hist):
    pos_rev = pos_hist[::-1].astype(np.float64)
    neg_rev = neg_hist[::-1].astype(np.float64)
    tp = np.cumsum(pos_rev)
    fp = np.cumsum(neg_rev)
    total_pos = float(pos_hist.sum())
    total_neg = float(neg_hist.sum())

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(total_pos, 1.0)
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))

    if total_pos == 0 or total_neg == 0:
        auroc = 0.0
    else:
        tpr = np.concatenate(([0.0], tp / total_pos))
        fpr = np.concatenate(([0.0], fp / total_neg))
        auroc = float(np.trapz(tpr, fpr))
    return ap, auroc


def main():
    opt = parse_args()
    thresholds = [float(v) for v in opt.thresholds.split(",") if v.strip()]
    hypes = prepare_hypes(opt)

    print("Creating Model")
    model = train_utils.create_model(hypes)
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if hasattr(model, "forward_colla"):
        model.force_collab = True
    model.eval()
    print(f"resume from {resume_epoch} epoch.")

    print("Dataset Building")
    dataset = build_dataset(hypes, visualize=False, train=False)
    eval_dataset = (
        Subset(dataset, range(min(opt.max_samples, len(dataset))))
        if opt.max_samples is not None
        else dataset
    )
    data_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    threshold_stats = {thr: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for thr in thresholds}
    pos_hist = np.zeros(opt.bins, dtype=np.int64)
    neg_hist = np.zeros(opt.bins, dtype=np.int64)
    total_abs = 0.0
    total_sq = 0.0
    total_pixels = 0
    total_pos = 0
    total_maps = 0
    skipped = 0

    for i, batch_data in enumerate(data_loader):
        if batch_data is None:
            skipped += 1
            continue
        if not opt.quiet:
            print(f"blindmap_eval_epoch{resume_epoch}_{i}", flush=True)

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data["ego"])
            pred = output_dict.get("pred_blind_maps", None)
            gt = batch_data["ego"].get("blind_maps_gt", None)
            if pred is None or gt is None:
                raise RuntimeError("Model output or batch data does not contain blind maps.")

            pred = collect_non_ego_predictions(pred, batch_data["ego"]["record_len"])
            if pred is None:
                skipped += 1
                continue
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            gt = gt.float()
            gt = (gt > 0.5).float()
            if gt.shape[-2:] != pred.shape[-2:]:
                gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")
            if gt.shape[0] != pred.shape[0]:
                raise RuntimeError(
                    f"BlindMap count mismatch: pred {tuple(pred.shape)}, gt {tuple(gt.shape)}"
                )

            pred = pred.float().clamp(0.0, 1.0)
            total_maps += pred.shape[0]
            total_pixels += pred.numel()
            total_pos += int(gt.sum().item())
            total_abs += torch.abs(pred - gt).sum().item()
            total_sq += torch.square(pred - gt).sum().item()

            for thr in thresholds:
                tp, fp, fn, tn = binary_counts(pred, gt, thr)
                threshold_stats[thr]["tp"] += int(tp)
                threshold_stats[thr]["fp"] += int(fp)
                threshold_stats[thr]["fn"] += int(fn)
                threshold_stats[thr]["tn"] += int(tn)

            bin_idx = torch.clamp((pred * (opt.bins - 1)).long(), 0, opt.bins - 1)
            gt_bool = gt.bool()
            pos_bins = bin_idx[gt_bool].detach().cpu()
            neg_bins = bin_idx[~gt_bool].detach().cpu()
            pos_hist += np.bincount(pos_bins.numpy(), minlength=opt.bins)
            neg_hist += np.bincount(neg_bins.numpy(), minlength=opt.bins)

    ap, auroc = hist_curves(pos_hist, neg_hist)
    pos_ratio = safe_div(total_pos, total_pixels)
    mae = safe_div(total_abs, total_pixels)
    brier = safe_div(total_sq, total_pixels)

    lines = [
        f"Epoch: {resume_epoch}",
        f"Range: {opt.range}",
        f"Samples: {len(eval_dataset)} | evaluated_maps: {total_maps} | skipped_batches: {skipped}",
        f"positive_pixel_ratio: {pos_ratio:.8f}",
        f"BlindMap AP/AUPRC: {ap:.6f}",
        f"BlindMap AUROC: {auroc:.6f}",
        f"MAE: {mae:.6f}",
        f"Brier: {brier:.6f}",
    ]
    for thr in thresholds:
        stat = threshold_stats[thr]
        tp, fp, fn, tn = stat["tp"], stat["fp"], stat["fn"], stat["tn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        iou = safe_div(tp, tp + fp + fn)
        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        specificity = safe_div(tn, tn + fp)
        lines.append(
            "thr={:.4f} | IoU: {:.6f} | Dice/F1: {:.6f} | Precision: {:.6f} | "
            "Recall: {:.6f} | Specificity: {:.6f}".format(
                thr, iou, dice, precision, recall, specificity
            )
        )

    msg = "\n".join(lines)
    print(msg)
    out_path = os.path.join(opt.model_dir, "blindmap_quality_result.txt")
    with open(out_path, "a+", encoding="utf-8") as f:
        f.write(msg + "\n\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
