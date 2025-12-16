import os
import argparse
from datetime import datetime
import time
import torch
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
import pascal_dataloader
import tools


def save_checkpoint(model, optimizer, scheduler, epoch, loss, global_step, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'global_step': global_step,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[Checkpoint] saved at epoch {epoch+1}: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv1 Training")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="實驗名稱；若不指定則用日期時間自動生成")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_name = args.exp_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    EPOCHS = 135
    BASE_LR = 5e-3   # 主體階段 LR（batch 64 時更保守，搭配 warmup）
    TRAIN_TXT = "/home/natsu/dataset/VOC0712_merged/train.txt"
    VAL_TXT   = "/home/natsu/dataset/VOC0712_merged/test.txt"
    BS_TRAIN = 16     # 實際 batch
    BS_VAL   = 1     # evaluate_map 假設 1
    map_interval = 150          # 每 N 個 epoch 評估一次 mAP
    checkpoint_interval = 10    # 每 N 個 epoch 儲存 checkpoint
    IMG_SIZE = 448
    ACCUM_STEPS = 4  # 梯度累積步數（有效 batch ≈ BS_TRAIN * ACCUM_STEPS）
    # 模型 head 目前只接受輸入 224 或 448（feature map 7x7 或 14x14），故先關閉多尺度
    MULTI_SCALES = None
    CUTOUT_PROB = 0.6
    CUTOUT_NUM_RANGE = (1, 3)
    CUTOUT_SIZE = (0.05, 0.2)
    # 等比例縮放 LR：目標有效 batch ~64，原 base LR 0.01 → scale = (BS*ACCUM)/64
    LR_SCALE = (BS_TRAIN * ACCUM_STEPS) / 64.0
    LR = BASE_LR * LR_SCALE

    # 實驗專用路徑
    ckpt_dir = os.path.join("checkpoints", exp_name)
    log_dir = os.path.join("runs", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    RESUME_PATH = os.path.join(ckpt_dir, "last.pth")

    train_transform = Compose([
        tools.BGR2RGB(),
        ToTensor(),
        ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        tools.BGR2RGB(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = pascal_dataloader.PascalDataset(
        txt_sample_path=TRAIN_TXT,
        transform=train_transform,
        img_size=IMG_SIZE,
        hflip_prob=0.5,
        affine_prob=0.5,
        multi_scale_sizes=MULTI_SCALES,
        cutout_prob=CUTOUT_PROB,
        cutout_num_range=CUTOUT_NUM_RANGE,
        cutout_size=CUTOUT_SIZE
    )
    train_loader = train_dataset.create_dataloader(batch_size=BS_TRAIN, shuffle=True, num_workers=16)

    val_dataset = pascal_dataloader.PascalDataset(
        txt_sample_path=VAL_TXT,
        transform=val_transform,
        img_size=IMG_SIZE,
        hflip_prob=0.0,
        affine_prob=0.0   # 驗證不做增強
    )
    val_loader = val_dataset.create_dataloader(batch_size=BS_VAL, shuffle=False, num_workers=8)

    net = model.YOLOv1(backbone_name="resnet50").to(device)
    criterion = model.YOLOv1Loss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 100], gamma=0.1)
    warmup_epochs = 3

    start_epoch = 0
    global_step = 0
    if os.path.isfile(RESUME_PATH):
        print(f"[Resume] Loading checkpoint from {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] start at epoch {start_epoch}, global_step {global_step}")

    head = model.YOLOv1Head(orig_img_size=(IMG_SIZE, IMG_SIZE),
                            iou_threshold=0.5,
                            scores_threshold=0.0)  # head 沒參數，不用放 device

    writer = SummaryWriter(log_dir=log_dir)
    train_start_time = time.time()
    # 若從 checkpoint 恢復，global_step 已在上方載入
    for epoch in range(start_epoch, EPOCHS):
        # 論文沒有 warmup，但用較大 LR + 448 輸入時，加個簡單 warmup 稳定訓練（僅從頭訓練時啟用）
        if start_epoch == 0 and epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = LR * lr_scale
        net.train()
        epoch_loss_sum = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
        for batch_idx, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device).float()

            preds = net(images)
            loss = criterion(preds, labels) / ACCUM_STEPS

            loss.backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss_sum += loss.item() * ACCUM_STEPS
            writer.add_scalar("train/loss_step", loss.item() * ACCUM_STEPS, global_step)
            global_step += 1
            # 更新 tqdm 狀態
            pbar.set_postfix({
                "loss": f"{loss.item() * ACCUM_STEPS:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.4f}"
            })

        avg_loss = epoch_loss_sum / len(train_loader)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")
        scheduler.step()
        val_map = None
        if (epoch + 1) % map_interval == 0:
            # mAP@0.5，用 head 的輸出
            val_map = tools.evaluate_map(
                model=net,
                head=head,
                dataloader=val_loader,
                device=device,
                num_classes=20,
                img_size=IMG_SIZE,
                iou_thr=0.5
            )
            writer.add_scalar("val/mAP@0.5", val_map, epoch)
            print(f"[Epoch {epoch+1}/{EPOCHS}] Val mAP@0.5: {val_map:.4f}")

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(net, optimizer, scheduler, epoch, avg_loss, global_step, ckpt_path)
            save_checkpoint(net, optimizer, scheduler, epoch, avg_loss, global_step, RESUME_PATH)

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()
