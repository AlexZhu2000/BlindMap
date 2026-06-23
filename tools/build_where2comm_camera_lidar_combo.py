import json
import shutil
from pathlib import Path

import torch
import yaml


ROOT = Path("/home/zzh/projects/BlindMap")
CAM_DIR = ROOT / "opencood/logs/Where2comm_opv2v_camera_pyramid_fair_2026_06_12_21_26_41_thre_0.01_add_noise_use_history"
LIDAR_DIR = ROOT / "opencood/logs/HeterBaseline_opv2v_lidar_pyramid_2026_05_30_16_11_39"
OUT_DIR = ROOT / "opencood/logs/Where2comm_opv2v_camera_lidar_combo_2026_06_17"


def load_yaml(path):
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def save_yaml(data, path):
    with path.open("w") as f:
        yaml.dump(data, f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cam_cfg = load_yaml(CAM_DIR / "config.yaml")
    lidar_cfg = load_yaml(LIDAR_DIR / "config.yaml")

    combo = cam_cfg
    combo["name"] = "Where2comm_opv2v_camera_lidar_combo"
    combo["input_source"] = ["lidar", "camera", "depth"]
    combo["label_type"] = "lidar"
    combo["use_history"] = False
    combo["history_num"] = 1
    combo["cav_lidar_range"] = lidar_cfg["cav_lidar_range"]
    combo["model"]["core_method"] = "where2comm_pyramid_collab_v2xset"
    combo["model"]["args"]["lidar_range"] = lidar_cfg["model"]["args"]["lidar_range"]
    combo["model"]["args"]["m1"] = lidar_cfg["model"]["args"]["m1"]
    combo["heter"]["ego_modality"] = "m1&m2"
    combo["heter"]["mapping_dict"] = {"m1": "m1", "m2": "m1", "m3": "m2", "m4": "m2"}
    combo["heter"]["modality_setting"]["m1"] = lidar_cfg["heter"]["modality_setting"]["m1"]
    combo["heter"]["modality_setting"]["m2"] = cam_cfg["heter"]["modality_setting"]["m2"]
    combo["model"]["args"]["m2"]["camera_mask_args"]["cav_lidar_range"] = lidar_cfg["cav_lidar_range"]
    combo["postprocess"]["anchor_args"]["cav_lidar_range"] = lidar_cfg["postprocess"]["anchor_args"]["cav_lidar_range"]
    combo["postprocess"]["gt_range"] = lidar_cfg["postprocess"]["gt_range"]
    combo["preprocess"] = lidar_cfg["preprocess"]
    combo["root_dir"] = lidar_cfg["root_dir"]
    combo["validate_dir"] = lidar_cfg["validate_dir"]
    combo["test_dir"] = lidar_cfg["test_dir"]
    combo["noise_setting"] = lidar_cfg["noise_setting"]
    combo["loss"] = lidar_cfg["loss"]
    combo["train_params"] = lidar_cfg["train_params"]
    combo["model"]["args"]["fusion_backbone"]["communication"]["use_threshold"] = True
    combo["model"]["args"]["fusion_backbone"]["communication"]["thre"] = 0.01
    combo["model"]["args"]["fusion_backbone"]["communication"]["comm_volume_MB"] = 1
    combo["model"]["args"]["fusion_backbone"]["comm_volume_MB"] = 1

    save_yaml(combo, OUT_DIR / "config.yaml")

    cam_ckpt = torch.load(CAM_DIR / "net_epoch_bestval_at21.pth", map_location="cpu")
    lidar_ckpt = torch.load(LIDAR_DIR / "net_epoch_bestval_at17.pth", map_location="cpu")
    merged = dict(cam_ckpt)
    for key, value in lidar_ckpt.items():
        if key.startswith(("encoder_m1.", "backbone_m1.", "aligner_m1.")):
            merged[key] = value

    torch.save(merged, OUT_DIR / "net_epoch_bestval_at21.pth")

    metadata = {
        "camera_model_dir": str(CAM_DIR),
        "camera_checkpoint": "net_epoch_bestval_at21.pth",
        "lidar_model_dir": str(LIDAR_DIR),
        "lidar_checkpoint": "net_epoch_bestval_at17.pth",
        "merged_policy": "camera checkpoint supplies m2, where2comm fusion, shrink and heads; lidar checkpoint supplies m1 encoder/backbone/aligner",
    }
    (OUT_DIR / "combo_metadata.json").write_text(json.dumps(metadata, indent=2))
    if not (OUT_DIR / "result.txt").exists():
        (OUT_DIR / "result.txt").write_text("")

    for script_name in [
        "run_native_where2comm_opv2v_camera_lidar_combo_parallel.sh",
    ]:
        src = ROOT / "opencood/tools" / script_name
        if src.exists():
            shutil.copy2(src, OUT_DIR / script_name)

    print(OUT_DIR)


if __name__ == "__main__":
    main()
