
"""
Train + test multimodal fusion models (early / intermediate / late).


Expected dataset layout (inside --dataset-path):
  - outcomes.csv
      Must include: case_id, <outcome column>
  - clinical.csv
      Must include: case_id and clinical feature columns

Images (passed via --image-path) should point to a folder containing one .npy per patient.
Each .npy must contain the 8 channels (pre+post combined) with shape (8, D, H, W). The dataset will select 16 evenly spaced slices along depth.

"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import resnet10
from monai.transforms import Compose, RandFlip, RandRotate90, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from helper import (
    EightChannelNpyDataset,
    EarlyFusionModel,
    SimpleIntermediateFusionModel,
    LateFusionModel,
    SimpleClinicalClassifier,
    replicate_first_layer,
    train_fusion,
    validate_fusion,
    train_model_fusion,
)



def _load_inputs(dataset_path: str, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned CSVs expected by this script.
    Returns: outcomes_df, clinical_df
    """
    dataset_path = os.path.abspath(dataset_path)
    outcomes_fp = os.path.join(dataset_path, "outcomes.csv")
    clinical_fp = os.path.join(dataset_path, "clinical.csv")

    missing = [p for p in [outcomes_fp, clinical_fp] if not os.path.exists(p)]
    if missing:
        msg = (
            "Missing required files under --dataset-path.\n"
            "Expected:\n"
            f"  {outcomes_fp}\n  {clinical_fp}\n\n"
            "Missing:\n  " + "\n  ".join(missing)
        )
        raise FileNotFoundError(msg)

    outcomes_df = pd.read_csv(outcomes_fp)
    if "case_id" not in outcomes_df.columns or outcome not in outcomes_df.columns:
        raise ValueError(f"outcomes.csv must contain columns: case_id and '{outcome}'")

    clinical_df = pd.read_csv(clinical_fp)
    if "case_id" not in clinical_df.columns:
        raise ValueError("clinical.csv must contain a 'case_id' column (same as notebooks).")

    return outcomes_df, clinical_df

def _make_transforms() -> tuple[Compose, Compose]:
    train_transforms = Compose(
        [
            Resize([16, 128, 128]),
            RandFlip(prob=0.5, spatial_axis=[0]),
            RandFlip(prob=0.5, spatial_axis=[1]),
            RandRotate90(prob=0.5, spatial_axes=[1, 2]),
            ToTensor(),
        ]
    )
    test_transforms = Compose([Resize([16, 128, 128]), ToTensor()])
    return train_transforms, test_transforms


def _split(outcomes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split 
      - 80/20 train/test (random_state=11)
      - train split again into train/train2 (random_state=32), but script uses train+val+test only.
    """
    from sklearn.model_selection import train_test_split

    pts_train, pts_test = train_test_split(outcomes_df, test_size=0.2, random_state=11)
    pts_train, pts_val = train_test_split(pts_train, test_size=0.2, random_state=32)
    return pts_train, pts_val, pts_test


def _build_model(
    fusion_type: str,
    base_model,
    input_size: int,
    imaging_weights_path: str | None,
    clinical_weights_path: str | None,
) -> torch.nn.Module:
    if fusion_type == "early":
        # notebook: EarlyFusionModel(pre_resnet10, input_size=len(train_dataset.columns), imaging_weights_path=None)
        return EarlyFusionModel(base_model, input_size=input_size, imaging_weights_path=imaging_weights_path)
    if fusion_type == "intermediate":
        # notebook: SimpleIntermediateFusionModel(pre_resnet10, input_size=len(train_dataset.columns), clinical_weights_path=...)
        return SimpleIntermediateFusionModel(
            base_model, input_size=input_size, clinical_weights_path=clinical_weights_path, imaging_weights_path=imaging_weights_path
        )
    if fusion_type == "late":
        # notebook: LateFusionModel(pre_resnet10, clinical_model, clinical_weights_path=..., imaging_weights_path=...)
        clinical_model = SimpleClinicalClassifier(input_size=input_size, fc_size=64)
        return LateFusionModel(
            base_model,
            clinical_model,
            imaging_weights_path=imaging_weights_path,
            clinical_weights_path=clinical_weights_path,
        )
    raise ValueError(f"Unknown fusion_type: {fusion_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Folder containing CSVs (outcomes.csv, clinical.csv)",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Folder containing one 8-channel .npy per patient.",
    )
    parser.add_argument("--fusion-type", required=True, choices=["early", "intermediate", "late"])
    parser.add_argument("--outcome", default="Margin_Binary")
    parser.add_argument(
        "--image-folder",
        default="",
        help="Optional subfolder inside --image-path where the .npy files live.",
    )
    parser.add_argument(
        "--image-filename-template",
        default="{case_id}.npy",
        help="Filename pattern for each case_id.",
    )
    parser.add_argument("--include-image", action="store_true", default=True)
    parser.add_argument("--no-include-image", dest="include_image", action="store_false")

    # Training hyperparameters (defaults chosen to mirror notebook final training sections)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--n-fold", type=int, default=3)
    parser.add_argument("--run-name", default=None, help="If omitted, a default name based on fusion-type + timestamp is used.")

    # Model weights
    parser.add_argument("--pretrained-resnet-weights", default=None, help="Path to MedicalNet resnet10 weights used by replicate_first_layer (optional).")
    parser.add_argument("--imaging-weights-path", default=None)
    parser.add_argument("--clinical-weights-path", default=None)

    # System
    parser.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu" (default: auto)')

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.fusion_type}_fusion_{current_time}"

    outcomes_df, clinical_df = _load_inputs(args.dataset_path, args.outcome)

    pts_train, pts_val, pts_test = _split(outcomes_df)

    train_transforms, test_transforms = _make_transforms()

    train_dataset = EightChannelNpyDataset(
        pts_train,
        args.outcome,
        clinical_df,
        transform=train_transforms,
        image_folder=args.image_folder,
        filename_template=args.image_filename_template,
        root=args.image_path,
        include_image=args.include_image,
    )
    val_dataset = EightChannelNpyDataset(
        pts_val,
        args.outcome,
        clinical_df,
        transform=test_transforms,
        image_folder=args.image_folder,
        filename_template=args.image_filename_template,
        root=args.image_path,
        include_image=args.include_image,
    )
    test_dataset = EightChannelNpyDataset(
        pts_test,
        args.outcome,
        clinical_df,
        transform=test_transforms,
        image_folder=args.image_folder,
        filename_template=args.image_filename_template,
        root=args.image_path,
        include_image=args.include_image,
    )

    # Base imaging model (same as notebooks: MONAI resnet10, 3D, 8 channels, 1 class)
    base_model = resnet10(pretrained=False, spatial_dims=3, n_input_channels=8, num_classes=1)
    if args.pretrained_resnet_weights:
        base_model = replicate_first_layer(args.pretrained_resnet_weights, base_model)

    input_size = len(train_dataset.columns)  # notebook uses len(train_dataset.columns)

    model = _build_model(
        args.fusion_type,
        base_model,
        input_size=input_size,
        imaging_weights_path=args.imaging_weights_path,
        clinical_weights_path=args.clinical_weights_path,
    )

    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "model": "resnet-10",
        "n_fold": args.n_fold,
        "run_name": run_name,
    }

    # Train with helper.train_model_fusion (same CV logic as notebooks)
    train_out = train_model_fusion(config, train_dataset, model, early_stopping=False, current_time=current_time, device=device)

    # Test evaluation (keep same helper.validate_fusion call shape)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(train_out["model"].parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logdir = f"Logs/{current_time}/{run_name}_FINAL"
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    writers = [writer] * max(1, args.n_fold)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # threshold comes from train_fusion; if absent, default 0.5
    threshold = 0.5
    try:
        # one quick pass to obtain a threshold from train set like notebooks do (via train_fusion)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        quick_train = train_fusion(train_out["model"], optimizer, criterion, train_loader, 0, writers, writer, 0, device)
        threshold = quick_train.get("threshold", 0.5)
    except Exception:
        threshold = 0.5

    test_metrics = validate_fusion(train_out["model"], optimizer, criterion, test_loader, 0, writers, writer, 0, threshold, device, True)

    # Save trained model
    out_dir = os.path.join(os.path.abspath(args.dataset_path), "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}_{args.fusion_type}.pth")
    torch.save(train_out["model"].state_dict(), out_path)

    print("\n=== TRAIN (CV) SUMMARY ===")
    print(train_out)
    print("\n=== TEST METRICS ===")
    print(test_metrics)
    print(f"\nSaved model: {out_path}")


if __name__ == "__main__":
    main()
