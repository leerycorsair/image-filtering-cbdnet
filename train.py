import os
import argparse
import torch
import torch.nn as nn

from dataset.loader import Real
from methods.models.my_cnn_model import MyCNNModel, fixed_loss
from consts import PATHS, NAMES


def train(train_loader, model, criterion, optimizer, scheduler):
    model.train()
    for noise_img, clean_img in train_loader:
        input_var = noise_img.cpu()
        target_var = clean_img.cpu()
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()


def main(batch_size: int, epochs: int, lr: float) -> None:
    save_path = PATHS["models"]
    train_path = PATHS["train_dataset"]
    model_name = NAMES["my_cnn"]
    model = MyCNNModel()
    model.cpu()
    model = nn.DataParallel(model)
    if os.path.exists(os.path.join(save_path, model_name)):
        model_info = torch.load(
            os.path.join(save_path, model_name), map_location=torch.device("cpu")
        )
        print("==> loading existing model:", os.path.join(save_path, model_name))
        model.load_state_dict(model_info["state_dict"])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler.load_state_dict(model_info["scheduler"])
        cur_epoch = model_info["epoch"]
    else:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        cur_epoch = 0

    criterion = fixed_loss()
    criterion.cpu()

    train_dataset = Real(train_path, 5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(cur_epoch, cur_epoch + epochs + 1):
        train(train_loader, model, criterion, optimizer, scheduler)

        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(save_path, model_name),
        )

        print(
            "Epochs [{0}/{1}]\t"
            "lr: {lr:.10f}".format(epoch, cur_epoch + epochs, lr=optimizer.param_groups[-1]["lr"])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=32, type=int, help="Размер батча (по умолчанию 32)")
    parser.add_argument("--epochs", default=50, type=int, help="Количество эпох (по умолчанию 50)")
    parser.add_argument(
        "--lr", default=2e-4, type=float, help="Скорость обучения (по умолчанию 0.00002)"
    )
    args = parser.parse_args()

    main(args.bs, args.epochs, args.lr)
