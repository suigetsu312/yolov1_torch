import os
import torch    


def PathBind(path: str) -> str:
    """
    Expand ~ to home for cross-platform path strings.
    """
    if path.startswith("~"):
        return os.path.expanduser(path)
    return path

def grid(S: int = 7):
    h = torch.arange(0, S).view(1, -1).repeat(S, 1).T
    w = torch.arange(0, S).view(1, -1).repeat(S, 1)
    grid = torch.stack([w, h], dim=-1)
    grid = grid.view(1, S, S, 2).float()
    return grid

def decode_bbox_offsets(boxes_offset, grid_size: int = 7):
    """
    將 offset 形式的 box (tx, ty, sqrt(w), sqrt(h)) 轉成 xywh（中心座標與寬高，0~1）。
    boxes_offset: (B, S, S, Bbox, 4)
    回傳: xywh normalized, shape 同 boxes_offset
    """
    device = boxes_offset.device
    _, S, _, Bbox, _ = boxes_offset.shape
    g = grid(S).to(device)                      # (1,S,S,2)
    g = g.unsqueeze(3).expand(-1, -1, -1, Bbox, -1)  # (1,S,S,Bbox,2)

    xy_cell = boxes_offset[..., :2] + g
    xy = xy_cell / float(grid_size)
    wh = torch.square(boxes_offset[..., 2:])
    return torch.cat([xy, wh], dim=-1)

def calc_iou(boxes1, boxes2):
    """
    element-wise IoU
    boxes1, boxes2: (..., 4)  [x, y, w, h]，中心點 + 寬高
    回傳: (...,) 同樣的前綴 shape
    """
    # 1. 拆 xy, wh
    b1_xy = boxes1[..., :2]
    b1_wh = boxes1[..., 2:]
    b2_xy = boxes2[..., :2]
    b2_wh = boxes2[..., 2:]

    # 2. 轉成 (x1,y1,x2,y2)
    b1_wh_half = b1_wh / 2.0
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    b2_wh_half = b2_wh / 2.0
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    # 3. intersection
    inter_min = torch.max(b1_min, b2_min)
    inter_max = torch.min(b1_max, b2_max)
    inter_wh = torch.clamp(inter_max - inter_min, min=0.0)

    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    union_area = b1_area + b2_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    iou = torch.clamp(iou, min=0.0, max=1.0)
    return iou

def box_iou_matrix(box1, box2):
    """
    box1: (N,4), box2: (M,4) in x1y1x2y2
    回傳 IoU 矩陣: (N,M)
    """
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.size(0), box2.size(0)), device=box1.device)

    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def targets_to_boxes(y_true, S=7, B=2, num_classes=20, img_size=224, device=None):
    """
    將 label grid (B,S,S,30) 轉成 per-image bbox list。
    回傳: list 長度 = batch_size
      all_gt[b] = tensor(K, 6) -> [class, x1, y1, x2, y2, 0]
    """
    if device is None:
        device = y_true.device
    y_true = y_true.to(device)

    batch_size = y_true.size(0)
    C = num_classes

    cls = y_true[..., :C]                                  # (B,S,S,C)
    boxes_offset = y_true[..., C:C+4*B] \
                        .view(batch_size, S, S, B, 4)     # (B,S,S,B,4)
    conf = y_true[..., C+4*B:C+5*B]                       # (B,S,S,B)

    # grid: (1,S,S,2) -> (B,S,S,B,2)
    g = grid().to(device)                                 # (1,S,S,2)
    g = g.unsqueeze(3)                                    # (1,S,S,1,2)
    g = g.expand(batch_size, S, S, B, 2)                  # (B,S,S,B,2)

    xy_cell = boxes_offset[..., :2] + g                   # (B,S,S,B,2)
    xy = xy_cell / float(S)                               # normalize 0~1
    wh = torch.square(boxes_offset[..., 2:])             # (B,S,S,B,2)  # labels 存 sqrt wh，這裡還原寬高

    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    boxes_norm = torch.cat([x1y1, x2y2], dim=-1)          # (B,S,S,B,4)

    all_gt = []
    for b in range(batch_size):
        gt_boxes = []
        for i in range(S):
            for j in range(S):
                # cell 裡完全沒物體就跳過
                if conf[b, i, j].sum() <= 0:
                    continue

                cls_id = torch.argmax(cls[b, i, j]).item()
                # 兩個 box GT 一樣，拿 box0 即可
                box = boxes_norm[b, i, j, 0] * img_size   # (4,)
                gt_boxes.append(torch.tensor([
                    float(cls_id),
                    box[0].item(),
                    box[1].item(),
                    box[2].item(),
                    box[3].item(),
                    0.0
                ], device=device))
        if len(gt_boxes) > 0:
            all_gt.append(torch.stack(gt_boxes, dim=0))   # (K,6)
        else:
            all_gt.append(torch.zeros((0, 6), device=device))
    return all_gt


def voc_ap(recalls, precisions):
    """
    Compute AP with the VOC 2010 style (integrate precision envelope).
    recalls, precisions: 1D tensors sorted by descending score.
    """
    device = recalls.device
    mrec = torch.cat([torch.tensor([0.0], device=device), recalls, torch.tensor([1.0], device=device)])
    mpre = torch.cat([torch.tensor([0.0], device=device), precisions, torch.tensor([0.0], device=device)])

    # Precision envelope
    for i in range(mpre.size(0) - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    # Summation over recall steps
    idx = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze()
    if idx.numel() == 0:
        return 0.0
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def compute_map_single_image(pred_boxes, gt_boxes, num_classes=20, iou_thr=0.5, device=None):
    """
    pred_boxes: (N,6) [cls, x1,y1,x2,y2, score]  ← 直接用 head 輸出
    gt_boxes:   (M,6) [cls, x1,y1,x2,y2,_]       ← decode label
    回傳: 該張圖的 mAP@iou_thr
    """
    if device is None:
        if pred_boxes.numel() > 0:
            device = pred_boxes.device
        else:
            device = gt_boxes.device

    if gt_boxes.numel() == 0:
        return 0.0

    aps = []
    for c in range(num_classes):
        gt_c = gt_boxes[gt_boxes[:, 0] == float(c)]
        pred_c = pred_boxes[pred_boxes[:, 0] == float(c)]

        if gt_c.size(0) == 0 and pred_c.size(0) == 0:
            continue
        if gt_c.size(0) == 0:
            aps.append(0.0)
            continue
        if pred_c.size(0) == 0:
            aps.append(0.0)
            continue

        n_gt = gt_c.size(0)

        # 按 score 排序
        scores = pred_c[:, -1]
        order = torch.argsort(scores, descending=True)
        pred_c = pred_c[order]

        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

        tps = []
        fps = []
        for k in range(pred_c.size(0)):
            p_box = pred_c[k, 1:5].unsqueeze(0)          # (1,4)
            ious = box_iou_matrix(p_box, gt_c[:, 1:5])   # (1,n_gt)
            iou_max, idx = ious.max(dim=1)
            if iou_max.item() >= iou_thr and not gt_matched[idx]:
                tps.append(1.0)
                fps.append(0.0)
                gt_matched[idx] = True
            else:
                tps.append(0.0)
                fps.append(1.0)

        tps = torch.tensor(tps, device=device)
        fps = torch.tensor(fps, device=device)
        cum_tp = torch.cumsum(tps, dim=0)
        cum_fp = torch.cumsum(fps, dim=0)
        recalls = cum_tp / (n_gt + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        # VOC integral AP (precision envelope)
        aps.append(voc_ap(recalls, precisions))

    if len(aps) == 0:
        return 0.0
    return sum(aps) / len(aps)


def evaluate_map(model, head, dataloader, device, num_classes=20,
                 img_size=224, iou_thr=0.5, return_per_class=False):
    """
    Pascal VOC 2007 樣式的 mAP@iou_thr（dataset-level）：
    - 將所有圖片的 pred/gt 彙整後，逐類別排序計算 TP/FP，再做 11-point AP。
    - batch_size 建議 1，若非 1 也會一張張處理。
    """
    model.eval()

    # 收集所有 GT/Pred，依類別存
    all_gt = {c: {} for c in range(num_classes)}      # cls -> {img_id: [boxes], matched flags}
    all_preds = {c: [] for c in range(num_classes)}   # cls -> [(score, img_id, box)]
    image_id = 0

    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            pred_boxes_batch = head(preds)

            # head 可能回傳 Tensor（batch=1）或 List[Tensor]（batch>1）
            if isinstance(pred_boxes_batch, torch.Tensor):
                pred_boxes_batch = [pred_boxes_batch]
            else:
                assert len(pred_boxes_batch) == batch_size, "Head output len != batch size"

            # decode GT
            gt_list = targets_to_boxes(
                labels, S=7, B=2,
                num_classes=num_classes,
                img_size=img_size,
                device=device
            )

            # 按 batch 逐張處理
            for b in range(batch_size):
                curr_img_id = image_id + b

                # GT
                gt_boxes = gt_list[b]
                for cls in range(num_classes):
                    gt_cls = gt_boxes[gt_boxes[:, 0] == float(cls)]
                    if gt_cls.numel() == 0:
                        continue
                    boxes_only = gt_cls[:, 1:5]
                    all_gt[cls][curr_img_id] = {
                        "boxes": boxes_only,
                        "matched": torch.zeros(len(boxes_only), dtype=torch.bool, device=device)
                    }

                # Pred
                pred_boxes = pred_boxes_batch[b]
                for row in pred_boxes:
                    cls = int(row[0].item())
                    score = row[-1].item()
                    box = row[1:5].to(device)
                    all_preds[cls].append((score, curr_img_id, box))

            image_id += batch_size

    ap_per_class = [0.0 for _ in range(num_classes)]

    # 逐類別計算 AP
    for cls in range(num_classes):
        preds = all_preds[cls]
        if len(preds) == 0:
            ap_per_class[cls] = 0.0
            continue
        # 排序
        preds = sorted(preds, key=lambda x: x[0], reverse=True)

        tp = []
        fp = []
        npos = sum(len(v["boxes"]) for v in all_gt[cls].values())
        if npos == 0:
            ap_per_class[cls] = 0.0
            continue

        for score, img_id, box in preds:
            gt_info = all_gt[cls].get(img_id, None)
            if gt_info is None:
                fp.append(1.0)
                tp.append(0.0)
                continue

            gt_boxes = gt_info["boxes"]
            matched = gt_info["matched"]

            ious = box_iou_matrix(box.unsqueeze(0), gt_boxes).squeeze(0)  # (n_gt,)
            iou_max, max_idx = (ious.max(0))

            if iou_max >= iou_thr and not matched[max_idx]:
                tp.append(1.0)
                fp.append(0.0)
                matched[max_idx] = True
            else:
                tp.append(0.0)
                fp.append(1.0)

        tp = torch.tensor(tp, device=device)
        fp = torch.tensor(fp, device=device)
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recalls = cum_tp / (npos + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        # VOC integral AP (precision envelope)
        ap_per_class[cls] = voc_ap(recalls, precisions)

    m_ap = sum(ap_per_class) / num_classes if num_classes > 0 else 0.0
    if return_per_class:
        return m_ap, ap_per_class
    return m_ap

class BGR2RGB:
    def __call__(self, x):
        return x[..., ::-1].copy()   # BGR → RGB
