import argparse
from pathlib import Path
import random

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

import model
from constants import CLASSES_VOC
import tools

# VOC 20 classes
CLASSES = CLASSES_VOC


def load_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cid, xc, yc, w, h = line.strip().split()
            cid = int(cid)
            xc, yc, w, h = map(float, (xc, yc, w, h))
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append({"cls": cid, "box": (x1, y1, x2, y2)})
    return boxes


def guess_label_path(img_path, txt_root=None):
    img_path = Path(img_path)
    name = img_path.stem
    candidates = []
    if txt_root:
        candidates.append(Path(txt_root) / "sample" / f"{name}.txt")
    candidates.append(img_path.with_suffix(".txt"))
    candidates.append(img_path.parent.parent / "sample" / f"{name}.txt")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def draw_boxes(img, boxes, color, prefix=""):
    for b in boxes:
        cid = b["cls"]
        x1, y1, x2, y2 = map(int, b["box"])
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"id{cid}"
        text = f"{prefix}{label}"
        if "score" in b:
            text += f" {b['score']:.2f}"
        cv.putText(img, text, (x1, max(0, y1 - 5)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return img


def build_transform():
    return Compose([
        tools.BGR2RGB(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])


def main():
    parser = argparse.ArgumentParser(description="Run model prediction on one image and visualize.")
    parser.add_argument("--image", default=None, help="圖片路徑；若未提供，需搭配 --txt 與 --index 從列表讀取。")
    parser.add_argument("--txt", default="/home/natsu/dataset/VOC0712_merged/test.txt",
                        help="train/test.txt 路徑（內含影像路徑列表），用來推斷 label 目錄。")
    parser.add_argument("--index", type=int, default=5243,
                        help="當提供 --txt 且未給 --image 時，從列表選第 index 行的圖片。")
    parser.add_argument("--label", default=None, help="標籤檔路徑；若未提供會自動猜測。")
    parser.add_argument("--checkpoint", default="checkpoints/last.pth", help="模型 checkpoint 路徑。")
    parser.add_argument("--img-size", type=int, default=448, help="輸入尺寸，會對圖片 resize。")
    parser.add_argument("--score-thr", type=float, default=0.2, help="分數門檻。")
    parser.add_argument("--iou-thr", type=float, default=0.45, help="NMS IoU 門檻。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="裝置，預設自動偵測。")
    parser.add_argument("--show", default=True, action="store_true", help="使用 cv2.imshow 顯示。")
    parser.add_argument("--out", default=None, help="若指定，將結果存檔。")
    args = parser.parse_args()

    txt_root = Path(args.txt).parent if args.txt else None
    if args.image:
        image_path = Path(args.image)
    else:
        if not args.txt:
            raise ValueError("未提供 --image，請搭配 --txt 與 --index 指定資料。")
        txt_path = Path(args.txt)
        if not txt_path.exists():
            raise FileNotFoundError(f"找不到 txt: {txt_path}")
        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if args.index < 0 or args.index >= len(lines):
            raise IndexError(f"index 超出範圍 0..{len(lines)-1}")
        image_path = Path(lines[args.index])

    img = cv.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"讀不到圖片: {image_path}")
    img = cv.resize(img, (args.img_size, args.img_size))
    h, w = img.shape[:2]

    label_path = Path(args.label) if args.label else guess_label_path(image_path, txt_root)
    gt_boxes = []
    if label_path and label_path.exists():
        print(f"Label directory: {label_path.parent}")
        gt_boxes = load_label(label_path, w, h)
        print(f"Loaded {len(gt_boxes)} GT boxes from {label_path}")
    else:
        print("No label found;只顯示預測框。")

    device = torch.device(args.device)
    net = model.YOLOv1().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    head = model.YOLOv1Head(orig_img_size=(args.img_size, args.img_size),
                            iou_threshold=args.iou_thr,
                            scores_threshold=args.score_thr)

    transform = build_transform()
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = net(inp)
        pred_boxes = head(preds)

    # head(batch=1) 可能回傳 Tensor 或 list，統一拿第一個
    if isinstance(pred_boxes, (list, tuple)):
        pred_boxes = pred_boxes[0]

    pred_list = []
    for row in pred_boxes:
        cid = int(row[0].item())
        x1, y1, x2, y2, score = row[1:].tolist()
        pred_list.append({"cls": cid, "box": (x1, y1, x2, y2), "score": score})

    print(f"Pred boxes (after NMS, score>{args.score_thr}): {len(pred_list)}")
    img_vis = img.copy()
    if gt_boxes:
        img_vis = draw_boxes(img_vis, gt_boxes, color=(0, 255, 0), prefix="gt:")
    if pred_list:
        img_vis = draw_boxes(img_vis, pred_list, color=(0, 0, 255), prefix="pred:")

    if args.show:
        cv.imshow("pred_vs_gt", img_vis)
        print("Press any key to close the window.")
        cv.waitKey(0)
        cv.destroyAllWindows()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), img_vis)
        print(f"Saved to {out_path}")
    elif not args.show:
        print("No --out specified and --show not used; nothing saved or displayed.")


if __name__ == "__main__":
    main()
