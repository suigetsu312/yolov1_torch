import argparse
from pathlib import Path
import random
from datetime import datetime

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
    h, w = img.shape[:2]
    for b in boxes:
        cid = b["cls"]
        x1, y1, x2, y2 = b["box"]
        # 避免框跑出圖外
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y2 = int(np.clip(y2, 0, h - 1))
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
    parser.add_argument("--index", type=int, default=None,
                        help="當提供 --txt 且未給 --image 時，從列表選第 index 行的圖片（優先於 --num）。")
    parser.add_argument("--num", type=int, default=3,
                        help="從 txt 列表隨機選擇 n 張圖片（當未指定 --image 且未指定 --index 時使用）。")
    parser.add_argument("--seed", type=int, default=0, help="隨機選取圖片時的種子。")
    parser.add_argument("--label", default=None, help="標籤檔路徑；若未提供會自動猜測。")
    parser.add_argument("--checkpoint", default="/home/natsu/project/DL/yolov1_torch/checkpoints/exp_20251216_193546/last.pth", help="模型 checkpoint 路徑。")
    parser.add_argument("--img-size", type=int, default=448, help="輸入尺寸，會對圖片 resize。")
    parser.add_argument("--score-thr", type=float, default=0.2, help="分數門檻。")
    parser.add_argument("--iou-thr", type=float, default=0.45, help="NMS IoU 門檻。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="裝置，預設自動偵測。")
    parser.add_argument("--show", action="store_true", help="使用 cv2.imshow 顯示。")
    parser.add_argument("--out-dir", default=None, help="輸出目錄；未指定則使用 outputs/debug_<timestamp>")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_root = Path(args.txt).parent if args.txt else None
    device = torch.device(args.device)
    net = model.YOLOv1().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    head = model.YOLOv1Head(orig_img_size=(args.img_size, args.img_size),
                            iou_threshold=args.iou_thr,
                            scores_threshold=args.score_thr)
    ckpt_name = Path(args.checkpoint).name

    transform = build_transform()

    def run_one(image_path: Path, out_path: Path):
        img_orig = cv.imread(str(image_path))
        if img_orig is None:
            print(f"[Warn] 讀不到圖片: {image_path}")
            return
        orig_h, orig_w = img_orig.shape[:2]
        img = cv.resize(img_orig, (args.img_size, args.img_size))

        label_path = Path(args.label) if args.label else guess_label_path(image_path, txt_root)
        gt_boxes = []
        if label_path and label_path.exists():
            gt_boxes = load_label(label_path, orig_w, orig_h)

        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = net(inp)
            pred_boxes = head(preds)

        if isinstance(pred_boxes, (list, tuple)):
            pred_boxes = pred_boxes[0]

        sx = float(orig_w) / float(args.img_size)
        sy = float(orig_h) / float(args.img_size)
        pred_list = []
        for row in pred_boxes:
            cid = int(row[0].item())
            x1, y1, x2, y2, score = row[1:].tolist()
            # 將框放大回原圖尺寸
            pred_list.append({"cls": cid,
                              "box": (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                              "score": score})

        img_vis = img_orig.copy()
        if gt_boxes:
            img_vis = draw_boxes(img_vis, gt_boxes, color=(0, 255, 0), prefix="gt:")
        if pred_list:
            img_vis = draw_boxes(img_vis, pred_list, color=(0, 0, 255), prefix="pred:")

        # 印出類別/分數方便檢查
        for b in pred_list:
            cls_name = CLASSES[b["cls"]] if 0 <= b["cls"] < len(CLASSES) else f"id{b['cls']}"
            print(f"pred {cls_name:12s} score {b['score']:.3f} box {b['box']}")
        for b in gt_boxes:
            cls_name = CLASSES[b["cls"]] if 0 <= b["cls"] < len(CLASSES) else f"id{b['cls']}"
            print(f"gt   {cls_name:12s} box {b['box']}")

        cv.imwrite(str(out_path), img_vis)
        print(f"Saved: {out_path} (pred {len(pred_list)} boxes, gt {len(gt_boxes)})")
        if args.show:
            cv.imshow(str(out_path.name), img_vis)
            cv.waitKey(0)

    # 決定要跑哪些圖片
    image_paths = []
    if args.image:
        image_paths = [Path(args.image)]
    else:
        if not args.txt:
            raise ValueError("未提供 --image，請搭配 --txt 與 --index/--num 指定資料。")
        txt_path = Path(args.txt)
        if not txt_path.exists():
            raise FileNotFoundError(f"找不到 txt: {txt_path}")
        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if args.index is not None:
            if args.index < 0 or args.index >= len(lines):
                raise IndexError(f"index 超出範圍 0..{len(lines)-1}")
            image_paths = [Path(lines[args.index])]
        else:
            random.seed(args.seed)
            picks = random.sample(lines, k=min(args.num, len(lines)))
            image_paths = [Path(p) for p in picks]

    for img_path in image_paths:
        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        run_one(img_path, out_path)

    # 記錄本次使用的 checkpoint
    with open(out_dir / "checkpoint.txt", "w") as f:
        f.write(str(args.checkpoint))

    if args.show:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
