import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter

import model
import pascal_dataloader
import tools


def build_transform(train=True):
    if train:
        return Compose([
            tools.BGR2RGB(),
            ToTensor(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    return Compose([
        tools.BGR2RGB(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

TRAIN_TXT = "/home/natsu/dataset/VOC2012/train.txt"

def main():
    parser = argparse.ArgumentParser(description="Quickly overfit a small subset to sanity-check training.")
    parser.add_argument("--train-txt", default=TRAIN_TXT, help="train.txt 路徑")
    parser.add_argument("--limit", type=int, default=1, help="使用前 N 張圖做過擬合測試。")
    parser.add_argument("--epochs", type=int, default=20, help="訓練 epoch 數。")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率。")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="權重衰減（過擬合測試預設關閉）。")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam", help="優化器。")
    parser.add_argument("--no-augment", action="store_true", help="關閉增強以便更快過擬合。")
    parser.add_argument("--img-size", type=int, default=448, help="輸入尺寸。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="裝置，預設自動偵測。")
    parser.add_argument("--vis-out", default="runs/overfit_vis.jpg",
                        help="訓練完後用第一張圖跑一次推論並存檔的路徑；設為空字串可跳過。")
    parser.add_argument("--score-thr", type=float, default=0.01, help="推論的 score 門檻。")
    parser.add_argument("--iou-thr", type=float, default=0.45, help="推論的 NMS IoU 門檻。")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = build_transform(train=not args.no_augment)
    dataset = pascal_dataloader.PascalDataset(
        txt_sample_path=args.train_txt,
        transform=transform,
        img_size=args.img_size,
        hflip_prob=0.0 if args.no_augment else 0.5
    )
    indices = list(range(min(args.limit, len(dataset))))
    subset = Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    net = model.YOLOv1().to(device)
    criterion = model.YOLOv1Loss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    net.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        term_sums = {"coord": 0.0, "object": 0.0, "noobj": 0.0, "class": 0.0, "total": 0.0}
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            preds = net(images)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            terms = getattr(criterion, "last_terms", None)
            if terms:
                for k in term_sums.keys():
                    term_sums[k] += terms[k].item()
        avg_loss = epoch_loss / len(loader)
        avg_terms = {k: v / len(loader) for k, v in term_sums.items()}
        print(f"[Overfit Epoch {epoch+1}/{args.epochs}] "
              f"loss: {avg_loss:.4f} | "
              f"coord {avg_terms['coord']:.4f} "
              f"obj {avg_terms['object']:.4f} "
              f"noobj {avg_terms['noobj']:.4f} "
              f"class {avg_terms['class']:.4f}")

    # ── 訓練完用第一張圖畫一次（避免還要存 checkpoint 再載入）
    if args.vis_out != "":
        first_idx = indices[0]
        img_path = Path(dataset.dataset[first_idx])
        label_path = Path(dataset.label_dir) / f"{img_path.stem}.txt"

        img = cv.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"讀不到圖片: {img_path}")
        img = cv.resize(img, (args.img_size, args.img_size))
        h, w = img.shape[:2]

        # 讀 GT 框
        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cid, xc, yc, bw, bh = line.strip().split()
                    cid = int(cid)
                    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    gt_boxes.append((cid, x1, y1, x2, y2))

        # 推論
        net.eval()
        head = model.YOLOv1Head(orig_img_size=(args.img_size, args.img_size),
                                iou_threshold=args.iou_thr,
                                scores_threshold=args.score_thr)
        transform_infer = build_transform(train=False)
        with torch.no_grad():
            inp = transform_infer(img).unsqueeze(0).to(device)
            pred_boxes = head(net(inp))

        if isinstance(pred_boxes, (list, tuple)):
            pred_boxes = pred_boxes[0]

        vis_img = img.copy()
        # 畫 GT
        for cid, x1, y1, x2, y2 in gt_boxes:
            cv.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.putText(vis_img, f"gt:{cid}", (int(x1), max(0, int(y1) - 5)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        # 畫 pred
        for row in pred_boxes:
            cid = int(row[0].item())
            x1, y1, x2, y2, score = row[1:].tolist()
            cv.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv.putText(vis_img, f"p{cid}:{score:.2f}", (int(x1), max(0, int(y1) - 5)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        out_path = Path(args.vis_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), vis_img)
        print(f"[Overfit] saved visualization to {out_path}")


if __name__ == "__main__":
    main()
