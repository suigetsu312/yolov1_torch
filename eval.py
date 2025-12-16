import argparse
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

import model
import pascal_dataloader
import tools
from constants import CLASSES_VOC


def build_transform():
    return Compose([
        tools.BGR2RGB(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv1 checkpoint mAP@0.5.")
    parser.add_argument("--txt", default="~/dataset/VOC0712_merged/test.txt",
                        help="包含影像路徑的 txt（例如 VOC 的 train/test 列表）。")
    parser.add_argument("--checkpoint", default="checkpoints/exp_20251216_193546/last.pth",
                        help="模型 checkpoint 路徑。")
    parser.add_argument("--img-size", type=int, default=448,
                        help="輸入尺寸，會對圖片 resize。")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="驗證 batch size，建議 1。")
    parser.add_argument("--score-thr", type=float, default=0.001,
                        help="Head 分數門檻（score = class_prob * conf）。")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="mAP 與 NMS 的 IoU 門檻。")
    parser.add_argument("--no-nms", action=argparse.BooleanOptionalAction, default=False,
                        help="關掉 NMS，只用 score 門檻篩選，用來看純 recall/score 分佈。true/false 可指定。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="裝置，預設自動偵測。")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    transform = build_transform()
    dataset = pascal_dataloader.PascalDataset(
        txt_sample_path=args.txt,
        transform=transform,
        img_size=args.img_size,
        hflip_prob=0.0  # 驗證不做增強
    )
    dataloader = dataset.create_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    net = model.YOLOv1().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    head = model.YOLOv1Head(
        orig_img_size=(args.img_size, args.img_size),
        iou_threshold=args.iou_thr,
        scores_threshold=args.score_thr,
        apply_nms=not args.no_nms
    )

    map_val, ap_per_class = tools.evaluate_map(
        model=net,
        head=head,
        dataloader=dataloader,
        device=device,
        num_classes=20,
        img_size=args.img_size,
        iou_thr=args.iou_thr,
        return_per_class=True
    )
    print(f"mAP@{args.iou_thr:.2f}: {map_val:.4f}")
    if ap_per_class is not None:
        for idx, ap in enumerate(ap_per_class):
            name = CLASSES_VOC[idx] if idx < len(CLASSES_VOC) else f"class{idx}"
            print(f"{name:12s} {ap:.4f}")

if __name__ == "__main__":
    main()
