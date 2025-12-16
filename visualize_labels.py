import argparse
from pathlib import Path
import random

import cv2 as cv
import numpy as np

# VOC 20 classes
from constants import CLASSES_VOC
CLASSES = CLASSES_VOC


def load_label(label_path, img_w, img_h):
    """
    讀取 YOLO 格式標籤: class_id x_center y_center width height (皆為 0~1)
    回傳 list[dict]: {"cls": int, "box": (x1,y1,x2,y2)}
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append({"cls": cid, "box": (x1, y1, x2, y2)})
    return boxes


def guess_label_path(img_path, txt_root=None):
    """
    嘗試從圖片檔名推斷 label 路徑，貼合 pascal_dataset 的邏輯：
      1) 若提供 txt_root（train.txt 所在目錄），使用 txt_root/sample/{name}.txt
      2) 同目錄下的 {name}.txt
      3) 上層目錄的 sample/{name}.txt
    """
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


def draw_boxes(img, boxes, label_font_scale=0.5):
    rng = random.Random(1234)
    colors = [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in CLASSES]
    for b in boxes:
        cid = b["cls"]
        x1, y1, x2, y2 = map(int, b["box"])
        color = colors[cid % len(colors)]
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"id{cid}"
        cv.putText(img, label, (x1, max(0, y1 - 5)),
                   cv.FONT_HERSHEY_SIMPLEX, label_font_scale, color, 1, cv.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on one image.")
    parser.add_argument("--image", help="圖片路徑；若未提供，需搭配 --txt 與 --index 從列表讀取。")
    parser.add_argument("--txt", help="train.txt 路徑（內含影像路徑列表），用來推斷 label 目錄。")
    parser.add_argument("--index", type=int, default=0,
                        help="當提供 --txt 且未給 --image 時，從列表選第 index 行的圖片。")
    parser.add_argument("--label", help="標籤路徑 (class xc yc w h in 0~1). 若未提供會自動猜測。")
    parser.add_argument("--out", default=None, help="若指定，將標註後的圖片存檔至該路徑。")
    parser.add_argument("--show", action="store_true", help="使用 cv2.imshow 直接顯示。")
    parser.add_argument("--img-size", type=int, default=None,
                        help="若指定，將圖片縮放為 img-size x img-size 再繪製。")
    args = parser.parse_args()

    # 若未提供 image，從 txt + index 取得
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

    if args.img_size is not None:
        img = cv.resize(img, (args.img_size, args.img_size))

    h, w = img.shape[:2]
    label_path = Path(args.label) if args.label else guess_label_path(image_path, txt_root)
    if label_path is None:
        raise FileNotFoundError("找不到標籤檔，請使用 --label 指定。")
    if not label_path.exists():
        raise FileNotFoundError(f"標籤檔不存在: {label_path}")

    print(f"Label directory: {label_path.parent}")
    boxes = load_label(label_path, w, h)
    print(f"Loaded {len(boxes)} boxes from {label_path}")
    img = draw_boxes(img, boxes)

    if args.show:
        cv.imshow("label_vis", img)
        print("Press any key to close the window.")
        cv.waitKey(0)
        cv.destroyAllWindows()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), img)
        print(f"Saved visualization to {out_path}")
    elif not args.show:
        print("No --out specified and --show not used; nothing saved or displayed.")


if __name__ == "__main__":
    main()
