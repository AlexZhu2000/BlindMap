import argparse
import copy
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare one collaborator's BlindMap under two ego vehicles."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--ego_a", type=str, default=None)
    parser.add_argument("--ego_b", type=str, default=None)
    parser.add_argument("--co_id", type=str, default=None)
    parser.add_argument("--split", choices=["validate", "test"], default="validate")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_history", action="store_true")
    return parser.parse_args()


def set_history(hypes, enabled):
    hypes["use_history"] = enabled
    try:
        hypes["model"]["args"]["fusion_backbone"]["blindmap"]["use_history"] = enabled
    except KeyError:
        pass


def scenario_index_from_sample(dataset, sample_idx):
    for i, end_idx in enumerate(dataset.len_record):
        if sample_idx < end_idx:
            return i
    raise ValueError(f"sample_idx={sample_idx} is outside dataset length")


def choose_default_triplet(hypes, sample_idx=None):
    probe_hypes = copy.deepcopy(hypes)
    probe_hypes.pop("forced_ego_id", None)
    probe_hypes.pop("keep_cav_ids", None)
    dataset = build_dataset(probe_hypes, visualize=False, train=False)
    history_num = int(probe_hypes.get("history_num", 0))

    if sample_idx is not None:
        scenario_idx = scenario_index_from_sample(dataset, sample_idx)
        cav_ids = [cav_id for cav_id in dataset.scenario_database[scenario_idx].keys() if int(cav_id) >= 0]
        if len(cav_ids) < 3:
            raise RuntimeError(f"sample_idx={sample_idx} has fewer than three non-negative CAVs: {cav_ids}")
        return sample_idx, cav_ids[0], cav_ids[1], cav_ids[2]

    for scenario_idx, scenario in dataset.scenario_database.items():
        cav_ids = [cav_id for cav_id in scenario.keys() if int(cav_id) >= 0]
        if len(cav_ids) < 3:
            continue

        prev_end = 0 if scenario_idx == 0 else dataset.len_record[scenario_idx - 1]
        scenario_len = dataset.len_record[scenario_idx] - prev_end
        local_idx = min(max(history_num, 0), scenario_len - 1)
        sample_idx = prev_end + local_idx
        return sample_idx, cav_ids[0], cav_ids[1], cav_ids[2]

    raise RuntimeError("No scenario with at least three non-negative CAVs was found.")


def count_history_hits(dataset, sample_idx, ego_id, co_id):
    if not getattr(dataset, "use_history", False):
        return {"enabled": False, "hits": 0, "expected": 0, "missing": []}

    scenario_idx = scenario_index_from_sample(dataset, sample_idx)
    scenario_database = dataset.scenario_database[scenario_idx]
    scenario_folder = dataset.scenario_folders[scenario_idx]
    timestamp_index = sample_idx if scenario_idx == 0 else sample_idx - dataset.len_record[scenario_idx - 1]
    curr_timestamp_key = dataset.return_timestamp_key(
        scenario_database, timestamp_index + dataset.tau
    )
    history_timestamps = dataset.get_history_timestamps(
        scenario_database, curr_timestamp_key, dataset.history_num
    )
    blindmap_folder = os.path.join(
        dataset.history_blindmap_dir, f"scenario_{os.path.basename(scenario_folder)}"
    )

    missing = []
    hits = 0
    for ts in history_timestamps:
        blindmap_file = os.path.join(
            blindmap_folder, f"ego_{ego_id}_agent_{co_id}_ts_{ts}.npy"
        )
        if os.path.exists(blindmap_file):
            hits += 1
        else:
            missing.append(blindmap_file)

    return {
        "enabled": True,
        "hits": hits,
        "expected": len(history_timestamps),
        "missing": missing,
    }


def build_single_sample(hypes, sample_idx, ego_id, keep_cav_ids):
    run_hypes = copy.deepcopy(hypes)
    run_hypes["forced_ego_id"] = str(ego_id)
    run_hypes["keep_cav_ids"] = [str(x) for x in keep_cav_ids]
    dataset = build_dataset(run_hypes, visualize=False, train=False)
    sample = dataset[sample_idx]
    batch = dataset.collate_batch_test([sample])
    if batch is None:
        raise RuntimeError(f"sample_idx={sample_idx} produced an empty batch")
    return dataset, batch


def infer_one(model, hypes, sample_idx, ego_id, keep_cav_ids, co_id, device):
    dataset, batch_cpu = build_single_sample(hypes, sample_idx, ego_id, keep_cav_ids)
    cav_id_list = [str(x) for x in batch_cpu["ego"]["cav_id_list"]]
    if str(co_id) not in cav_id_list:
        raise RuntimeError(f"co_id={co_id} not in cav_id_list={cav_id_list}")

    batch = train_utils.to_device(batch_cpu, device)
    with torch.no_grad():
        output = model(batch["ego"])

    pred_blind_maps = output.get("pred_blind_maps", None)
    if pred_blind_maps is None:
        pyramid_state = output.get("pyramid")
        raise RuntimeError(
            f"Model output has no pred_blind_maps. pyramid={pyramid_state}, keys={list(output.keys())}. "
            "This comparison requires the collaborative forward branch; check model.force_collab."
        )
    pred_blind_maps = pred_blind_maps.detach().cpu()
    co_idx = cav_id_list.index(str(co_id))
    co_map = pred_blind_maps[co_idx, 0].numpy()

    comm_cfg = hypes["model"]["args"]["fusion_backbone"]["communication"]
    if comm_cfg.get("use_threshold", True):
        threshold = float(comm_cfg["thre"])
        co_mask = (co_map > threshold).astype(np.float32)
        comm_policy = f"threshold>{threshold}"
    else:
        threshold = float(np.quantile(co_map, 0.99))
        co_mask = (co_map > threshold).astype(np.float32)
        comm_policy = f"fallback_top1pct>{threshold:.6f}"

    return {
        "dataset": dataset,
        "batch": batch_cpu,
        "cav_id_list": cav_id_list,
        "co_idx": co_idx,
        "co_map": co_map,
        "co_mask": co_mask,
        "comm_policy": comm_policy,
        "history": count_history_hits(dataset, sample_idx, str(ego_id), str(co_id)),
    }


def plot_comparison(result_a, result_b, ego_a, ego_b, co_id, out_png):
    map_a = result_a["co_map"]
    map_b = result_b["co_map"]
    mask_a = result_a["co_mask"]
    mask_b = result_b["co_mask"]
    diff = np.abs(map_a - map_b)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=220)
    panels = [
        (map_a, f"co {co_id} BlindMap | ego {ego_a}", "magma", 0, 1),
        (map_b, f"co {co_id} BlindMap | ego {ego_b}", "magma", 0, 1),
        (diff, "absolute probability difference", "viridis", 0, None),
        (mask_a, f"comm mask | ego {ego_a}", "gray", 0, 1),
        (mask_b, f"comm mask | ego {ego_b}", "gray", 0, 1),
        (np.abs(mask_a - mask_b), "absolute mask difference", "inferno", 0, 1),
    ]

    for ax, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(None, args)
    if args.split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    if args.no_history:
        set_history(hypes, False)

    if args.sample_idx is None or args.ego_a is None or args.ego_b is None or args.co_id is None:
        sample_idx, ego_a, ego_b, co_id = choose_default_triplet(hypes, args.sample_idx)
        args.sample_idx = sample_idx if args.sample_idx is None else args.sample_idx
        args.ego_a = ego_a if args.ego_a is None else args.ego_a
        args.ego_b = ego_b if args.ego_b is None else args.ego_b
        args.co_id = co_id if args.co_id is None else args.co_id

    keep_cav_ids_a = [str(args.ego_a), str(args.co_id)]
    keep_cav_ids_b = [str(args.ego_b), str(args.co_id)]
    output_dir = Path(args.output_dir or Path(args.model_dir) / "receiver_conditioned_blindmap")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    epoch, model = train_utils.load_saved_model(args.model_dir, model)
    model.force_collab = True
    model = model.to(device)
    model.eval()

    result_a = infer_one(
        model, hypes, args.sample_idx, args.ego_a, keep_cav_ids_a, args.co_id, device
    )
    result_b = infer_one(
        model, hypes, args.sample_idx, args.ego_b, keep_cav_ids_b, args.co_id, device
    )

    mask_a = result_a["co_mask"].astype(bool)
    mask_b = result_b["co_mask"].astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    metrics = {
        "model_dir": args.model_dir,
        "epoch": int(epoch),
        "sample_idx": int(args.sample_idx),
        "ego_a": str(args.ego_a),
        "ego_b": str(args.ego_b),
        "co_id": str(args.co_id),
        "cav_id_list_a": result_a["cav_id_list"],
        "cav_id_list_b": result_b["cav_id_list"],
        "co_index_a": int(result_a["co_idx"]),
        "co_index_b": int(result_b["co_idx"]),
        "comm_policy": result_a["comm_policy"],
        "mask_iou": float(intersection / union) if union > 0 else 1.0,
        "map_l1_mean": float(np.mean(np.abs(result_a["co_map"] - result_b["co_map"]))),
        "mask_changed_ratio": float(np.mean(mask_a != mask_b)),
        "selected_cells_a": int(mask_a.sum()),
        "selected_cells_b": int(mask_b.sum()),
        "history_a": result_a["history"],
        "history_b": result_b["history"],
    }

    stem = f"sample_{args.sample_idx}_co_{args.co_id}_ego_{args.ego_a}_vs_{args.ego_b}"
    out_png = output_dir / f"{stem}.png"
    out_npz = output_dir / f"{stem}.npz"
    out_json = output_dir / f"{stem}.json"

    plot_comparison(result_a, result_b, args.ego_a, args.ego_b, args.co_id, out_png)
    np.savez_compressed(
        out_npz,
        map_a=result_a["co_map"],
        map_b=result_b["co_map"],
        mask_a=result_a["co_mask"],
        mask_b=result_b["co_mask"],
    )
    out_json.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {out_png}")
    print(f"Saved arrays: {out_npz}")
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
