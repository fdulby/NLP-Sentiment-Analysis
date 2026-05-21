# -*- coding: utf-8 -*-
"""
BiLSTM + MaxPool training script.

It follows train.py's data, training, early-stopping, class-weight, and output logic.
The only structural change is:
    BiLSTM outputs at all time steps -> masked max pooling -> classifier
"""

import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import train as base_train


CONFIG: Dict[str, Any] = {
    "processed_data_dir": "processed_data",
    "train_file": "train.csv",
    "val_file": "val.csv",
    "test_file": "test.csv",
    "vocab_file": "vocab.json",
    "metadata_file": "metadata.json",
    "output_root": "maxpool",
    "run_name_keys": [
        "lr",
        "batch_size",
        "hidden_dim",
        "dropout",
        "max_len",
        "num_layers",
        "embed_dim",
        "use_class_weights",
        "early_stopping_patience",
    ],
    "seed": 42,
    "max_len": 64,
    "embed_dim": 64,
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.5,
    "epochs": 30,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "num_workers": 0,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 1e-4,
    "use_class_weights": True,
    "label_names": None,
}


class BiLSTMMaxPoolClassifier(base_train.BiLSTMClassifier):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x != self.pad_idx).sum(dim=1)
        layer_input = self.embedding(x)

        for layer_idx in range(self.num_layers):
            fw_outputs, _ = self._run_one_direction(
                layer_input, lengths, self.fw_cells[layer_idx], reverse=False
            )
            bw_outputs, _ = self._run_one_direction(
                layer_input, lengths, self.bw_cells[layer_idx], reverse=True
            )
            layer_input = torch.cat([fw_outputs, bw_outputs], dim=-1)

        # H: (batch, seq_len, 2*hidden_dim)
        h_all = layer_input
        pad_mask = (x == self.pad_idx).unsqueeze(-1)
        h_all = h_all.masked_fill(pad_mask, -1e9)

        pooled = torch.max(h_all, dim=1).values
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def run_one_experiment(cfg: Dict[str, Any], data_info: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    base_train.set_seed(cfg["seed"])
    device = base_train.get_device()
    run_name = base_train.make_run_name(cfg)
    run_dir = os.path.join(base_dir, cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    paths = {
        "best_model": os.path.join(run_dir, "best_model.pt"),
        "loss_curve": os.path.join(run_dir, "loss_curve.png"),
        "wrong": os.path.join(run_dir, "wrong.csv"),
    }

    word2id = data_info["word2id"]
    metadata = data_info["metadata"]
    label_names = metadata["label_names"]
    num_classes = int(metadata["num_classes"])
    pad_idx = word2id.get("<PAD>", 0)

    train_loader, val_loader, test_loader = base_train.create_loaders(data_info, cfg)
    class_weights = None
    if cfg["use_class_weights"]:
        class_weights = base_train.compute_class_weights(
            train_csv=data_info["paths"]["train_csv"],
            num_classes=num_classes,
            device=device,
        )
        print(
            "类别权重:",
            {label_names[idx]: round(float(weight), 4) for idx, weight in enumerate(class_weights.cpu())},
        )

    model = BiLSTMMaxPoolClassifier(
        vocab_size=len(word2id),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = cfg["epochs"]

    print(f"\n开始 MaxPool 实验: {run_name}")
    print(f"设备: {device}")
    print(f"结果目录: {run_dir}")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = base_train.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg["grad_clip"],
        )
        val_result = base_train.evaluate(model, val_loader, device, criterion)
        val_loss = float(val_result["loss"])
        val_acc = float(val_result["acc"])
        train_losses.append(float(train_loss))
        val_losses.append(val_loss)

        is_better = base_train.is_validation_better(
            val_acc=val_acc,
            val_loss=val_loss,
            best_val_acc=best_val_acc,
            best_val_loss=best_val_loss,
            min_delta=cfg["early_stopping_min_delta"],
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "word2id": word2id,
                    "hparams": {
                        "max_len": cfg["max_len"],
                        "embed_dim": cfg["embed_dim"],
                        "hidden_dim": cfg["hidden_dim"],
                        "num_layers": cfg["num_layers"],
                        "dropout": cfg["dropout"],
                        "num_classes": num_classes,
                        "pad_idx": pad_idx,
                        "use_maxpool": True,
                    },
                    "label_names": label_names,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "config": cfg,
                },
                paths["best_model"],
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"best_epoch={best_epoch} | "
            f"no_improve={epochs_without_improvement}/{cfg['early_stopping_patience']}"
        )

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"早停触发: 连续 {cfg['early_stopping_patience']} 轮验证集无提升，"
                f"停止在 epoch {epoch}，最佳 epoch 为 {best_epoch}。"
            )
            break

    checkpoint = torch.load(paths["best_model"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = base_train.evaluate(model, test_loader, device, criterion)
    test_summary = base_train.build_test_summary(test_result, label_names)
    wrong_count = base_train.save_wrong_predictions(
        test_csv=data_info["paths"]["test_csv"],
        eval_result=test_result,
        label_names=label_names,
        save_path=paths["wrong"],
    )
    base_train.plot_loss_curve(train_losses, val_losses, test_summary, paths["loss_curve"])

    print(
        f"MaxPool 实验完成: {run_name} | "
        f"best_val_acc={best_val_acc:.4f} | "
        f"test_acc={test_summary['accuracy']:.4f} | "
        f"错误样本数={wrong_count} | "
        f"stopped_early={stopped_early} | stop_epoch={stop_epoch}"
    )
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "best_val_acc": best_val_acc,
        "test_acc": test_summary["accuracy"],
        "wrong_count": wrong_count,
    }


def main() -> None:
    os.chdir(BASE_DIR)
    data_info = base_train.build_data_info(CONFIG, BASE_DIR)
    summary = run_one_experiment(CONFIG, data_info, BASE_DIR)
    print(
        f"\n完成: {summary['run_name']} | "
        f"test_acc={summary['test_acc']:.4f} | run_dir={summary['run_dir']}"
    )


if __name__ == "__main__":
    main()
