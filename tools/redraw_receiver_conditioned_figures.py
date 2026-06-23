import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


INFOCOM_DIR = Path(
    "/home/zzh/projects/InfoCom/opencood/logs/"
    "infocom_opv2v_20260527_164045/receiver_conditioned_infocom"
)
BLINDMAP_DIR = Path(
    "/home/zzh/projects/BlindMap/opencood/logs/"
    "BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history*/"
    "receiver_conditioned_blindmap"
)


def load_meta(path):
    return json.loads(path.read_text())


def center_crop_like_infocom(arr, target_ratio):
    height, width = arr.shape
    crop_height = max(1, min(height, int(round(width * target_ratio))))
    top = (height - crop_height) // 2
    return arr[top : top + crop_height]


def add_row_colorbar(fig, im, anchor_ax, label):
    divider = make_axes_locatable(anchor_ax)
    cax = divider.append_axes("right", size="3%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)
    cbar.set_label(label, fontsize=7, labelpad=3)
    return cbar


def style_axis(ax, title):
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#444444")


def plot_receiver_conditioned(arrays, meta, out_png, method_name, score_key_a, score_key_b, crop_ratio=None):
    score_a = arrays[score_key_a]
    score_b = arrays[score_key_b]
    mask_a = arrays["mask_a"]
    mask_b = arrays["mask_b"]

    if crop_ratio is not None:
        score_a = center_crop_like_infocom(score_a, crop_ratio)
        score_b = center_crop_like_infocom(score_b, crop_ratio)
        mask_a = center_crop_like_infocom(mask_a, crop_ratio)
        mask_b = center_crop_like_infocom(mask_b, crop_ratio)

    score_diff = np.abs(score_a - score_b)
    mask_diff = np.abs(mask_a - mask_b)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 3.9),
        dpi=260,
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.035, right=0.965, top=0.87, bottom=0.075, wspace=0.16, hspace=0.34)

    receiver_a = meta["ego_a"]
    receiver_b = meta["ego_b"]
    sender = meta["co_id"]

    score_panels = [
        (score_a, f"sender {sender} score | receiver {receiver_a}", "magma", 0, 1),
        (score_b, f"sender {sender} score | receiver {receiver_b}", "magma", 0, 1),
        (score_diff, "absolute score difference", "magma", 0, 1),
    ]
    mask_panels = [
        (mask_a, f"selected mask | receiver {receiver_a}", "gray", 0, 1),
        (mask_b, f"selected mask | receiver {receiver_b}", "gray", 0, 1),
        (mask_diff, "absolute mask difference", "gray", 0, 1),
    ]

    score_im = None
    for ax, (image, title, cmap, vmin, vmax) in zip(axes[0], score_panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
        if score_im is None:
            score_im = im
        style_axis(ax, title)

    mask_im = None
    for ax, (image, title, cmap, vmin, vmax) in zip(axes[1], mask_panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
        if mask_im is None:
            mask_im = im
        style_axis(ax, title)

    add_row_colorbar(fig, score_im, axes[0, 2], "score")
    add_row_colorbar(fig, mask_im, axes[1, 2], "mask")

    fig.suptitle(
        f"{method_name}: receiver-conditioned sender-region comparison",
        fontsize=10,
        y=0.965,
    )
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main():
    infocom_npz = INFOCOM_DIR / "sample_10_sender_2167_receiver_2149_vs_2158.npz"
    infocom_json = INFOCOM_DIR / "sample_10_sender_2167_receiver_2149_vs_2158.json"
    blindmap_npz = BLINDMAP_DIR / "sample_10_co_2167_ego_2149_vs_2158.npz"
    blindmap_json = BLINDMAP_DIR / "sample_10_co_2167_ego_2149_vs_2158.json"

    infocom_arrays = np.load(infocom_npz)
    infocom_meta = load_meta(infocom_json)
    plot_receiver_conditioned(
        infocom_arrays,
        infocom_meta,
        INFOCOM_DIR / "sample_10_sender_2167_receiver_2149_vs_2158_optimized.png",
        "InfoCom",
        "raw_a",
        "raw_b",
    )

    blindmap_arrays = np.load(blindmap_npz)
    blindmap_meta = load_meta(blindmap_json)
    infocom_ratio = infocom_arrays["raw_a"].shape[0] / infocom_arrays["raw_a"].shape[1]
    plot_receiver_conditioned(
        blindmap_arrays,
        blindmap_meta,
        BLINDMAP_DIR / "sample_10_co_2167_ego_2149_vs_2158_optimized.png",
        "BlindMap",
        "map_a",
        "map_b",
        crop_ratio=infocom_ratio,
    )


if __name__ == "__main__":
    main()
