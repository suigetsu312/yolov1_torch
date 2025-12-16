import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
import timm
import tools


class YOLOv1(nn.Module):
    def __init__(self, num_classes: int = 20, B: int = 2,
                 pretrained: bool = True, S: int = 7,
                 backbone_name: str = "darknet53.c2ns_in1k"):
        super().__init__()
        self.S = S
        self.B = B
        self.C = num_classes

        # 1) timm Darknet53 backbone，直接取最後一層 feature map（stride 32 → 14x14 for 448）
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )
        in_ch = self.backbone.feature_info.channels()[-1]

        # 2) 若輸入 448 → 最後一層會是 14x14，需壓成 7x7
        #    用 AvgPool2d/MaxPool2d 都可以，看你喜好
        self.down_to_grid = nn.AvgPool2d(kernel_size=2, stride=2)

        # 3) head：把 [B, C, 7, 7] → [B, B*5 + C, 7, 7]
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, self.C + 5 * self.B, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone 回傳 list，取最後一個階段的 feature
        feat = self.backbone(x)[0]       # (B, C, H, W)

        H, W = feat.shape[2], feat.shape[3]

        # 如果是 2*S x 2*S（例如 14x14），就壓成 SxS（7x7）
        if H == 2 * self.S and W == 2 * self.S:
            feat = self.down_to_grid(feat)
        elif H != self.S or W != self.S:
            # 其它情況直接爆錯，避免你以為是 7x7 結果不是
            raise RuntimeError(
                f"Unexpected feature map size {H}x{W}, expected {self.S}x{self.S} or {2*self.S}x{2*self.S}"
            )

        out = self.head(feat)            # (B, 30, 7, 7) if C=20,B=2,S=7
        out = out.permute(0, 2, 3, 1)    # (B, 7, 7, 30)
        return out

class YOLOv1Loss(torch.nn.Module):
    def __init__(self, batch_size=16, 
                 grid=7,
                 numOfClasses=20,
                 numOfBox=2,
                 lambda_coord=5, 
                 lambda_obj =1, 
                 lambda_noobj=0.5,
                 lambda_class=1, ):
        super(YOLOv1Loss, self).__init__()
        self.numOfBox = numOfBox
        self.numOfClasses = numOfClasses
        self.grid = grid
        self.batch_size = batch_size
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        
        self.indexOfBoxAfter = self.numOfClasses + (self.numOfBox * 4)
        self.indexOfBoxBefore = self.numOfClasses

    def coordinate_loss(self, pred_xywh, true_xywh):
        # pred_xywh, true_xywh 都是 (B, S, S, B, 4)，包含 [tx, ty, t_sqrt(w), t_sqrt(h)]
        
        # 1. XY Loss (L2 損失)
        xy_loss = (pred_xywh[..., :2] - true_xywh[..., :2]) ** 2
        xy_loss_sum = torch.sum(xy_loss, dim=-1) # (B, S, S, B)
        
        # 2. WH Loss (sqrt 形式的 L2 損失)
        wh_loss = (pred_xywh[..., 2:] - true_xywh[..., 2:]) ** 2
        wh_loss_sum = torch.sum(wh_loss, dim=-1) # (B, S, S, B)
        
        coord_raw_loss = xy_loss_sum + wh_loss_sum
        
        # 應用 lambda_coord = 5
        return self.lambda_coord * coord_raw_loss
    
    def object_loss(self, iou, pred_conf):
        # element-wise (B,S,S,B) 供 mask 後再彙總
        return self.lambda_obj * (pred_conf - iou) ** 2

    def noobject_loss(self, iou, pred_conf):
        # noobj 的目標是 0（貼近 YOLOv1 設定）
        return self.lambda_noobj * (pred_conf ** 2)

    def class_loss(self, pred_classes, true_classes):
        # 將 true_classes 從 one-hot 轉換為類別索引
        true_classes = torch.argmax(true_classes, dim=-1)  # 轉換為 (16, 7, 7)
        # 調整 pred_classes 的形狀為 (batch_size, num_classes, height, width)
        pred_classes = pred_classes.permute(0, 3, 1, 2)  # (16, 7, 7, 20) -> (16, 20, 7, 7)

        class_loss = self.lambda_class * nn.functional.cross_entropy(pred_classes, true_classes, reduction='none')
        return class_loss
    def forward(self, y_pred, y_true):
        
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, self.grid, self.grid, (self.numOfBox* 5)+self.numOfClasses )
        y_true = y_true.view(batch_size, self.grid, self.grid, (self.numOfBox* 5)+self.numOfClasses )

        # bbox offset: (B, S, S, B, 4)
        ypred_bbox_offset = y_pred[..., self.indexOfBoxBefore:self.indexOfBoxAfter] \
                                .view(batch_size, self.grid, self.grid, self.numOfBox, 4)
        # xy 使用線性輸出（貼近原 YOLOv1），wh 維持線性（sqrt wh）
        ytrue_bbox_offset = y_true[..., self.indexOfBoxBefore:self.indexOfBoxAfter] \
                                .view(batch_size, self.grid, self.grid, self.numOfBox, 4)

        # xy 轉成 0~1 的比例座標，wh 先平方回真實比例（論文使用 sqrt w/h）
        ypred_bbox = tools.decode_bbox_offsets(ypred_bbox_offset, grid_size=self.grid)
        ytrue_bbox = tools.decode_bbox_offsets(ytrue_bbox_offset, grid_size=self.grid)

        # 標註的物件存在與否（每 box 一個 conf 標籤）
        label_response = (y_true[..., self.indexOfBoxAfter:] > 0).float()  # (B,7,7,2)

        # conf 線性輸出，貼近原論文（confidence = Pr(obj)*IoU）
        conf_pred = y_pred[..., self.indexOfBoxAfter:]
        iou_between_pred_true_box = tools.calc_iou(ypred_bbox, ytrue_bbox).to(y_pred.device)
        # iou_between_pred_true_box: (B, 7, 7, 2)
        best_box = torch.argmax(iou_between_pred_true_box, dim=-1)  # shape: (B, 7, 7)

        # cell 是否有物體（論文一 cell 只有一個物體）
        has_obj = (label_response.sum(dim=-1, keepdim=True) > 0).float()  # (B,7,7,1)
        # 負責的 box：IoU 最大者且 cell 內有物體
        object_mask = torch.nn.functional.one_hot(best_box, num_classes=self.numOfBox).float() * has_obj  # (B,7,7,2)
        # 非負責：其餘 box（包含空 cell 全部 box），需學成 no-object
        noobject_mask = 1.0 - object_mask

        # 使用 raw offset 空間計算 coordinate loss（論文設定）
        coord_loss = torch.sum(self.coordinate_loss(ypred_bbox_offset, ytrue_bbox_offset) * object_mask, dim=3)
        # 逐 box 計算後再套 mask，避免把兩個 box 的損失混在一起
        object_loss = torch.sum(self.object_loss(iou_between_pred_true_box, conf_pred) * object_mask, dim=3)
        no_object_loss = torch.sum(self.noobject_loss(iou_between_pred_true_box, conf_pred) * noobject_mask, dim=3)

        pred_class = y_pred[..., :self.indexOfBoxBefore]
        true_class = y_true[..., :self.indexOfBoxBefore]
        # 類別使用 softmax 後的 MSE，僅在有物體的 cell 上計算（論文設定）
        cell_obj_mask = (label_response.sum(dim=-1) > 0).float()      # (B,S,S)
        pred_class_prob = torch.softmax(pred_class, dim=-1)
        class_loss = self.lambda_class * torch.sum((pred_class_prob - true_class) ** 2, dim=-1)
        class_loss = class_loss * cell_obj_mask                    # (B,S,S)

        loss = torch.sum(coord_loss + object_loss + no_object_loss + class_loss, dim=[1, 2]) 
        # 紀錄 loss 組成方便除錯
        self.last_terms = {
            "coord": coord_loss.mean().detach(),
            "object": object_loss.mean().detach(),
            "noobj": no_object_loss.mean().detach(),
            "class": class_loss.mean().detach(),
            "total": loss.mean().detach(),
        }
        return loss.mean()
    
class YOLOv1Head(nn.Module):
    def __init__(self, 
                 orig_img_size,
                 grid=7,
                 numOfBox=2,
                 class_num=20,
                 iou_threshold=0.5,
                 scores_threshold=0.05,
                 apply_nms=True,
                 yolo_img_size=(224, 224),
                 name='yolov1_head'):
        super(YOLOv1Head, self).__init__()
        self.numOfBox = numOfBox
        self.grid = grid
        self.iou_threshold = iou_threshold
        self.scores_threshold = scores_threshold
        self.apply_nms = apply_nms
        self.yolo_img_size = torch.tensor(yolo_img_size, dtype=torch.float32)
        self.orig_img_size = torch.tensor(orig_img_size, dtype=torch.float32)
        self.class_num = class_num

    @staticmethod
    def nms(boxes, iou_threshold, scores_threshold):
        pred_box = boxes[:, 1:5]
        pred_scores = boxes[:, -1]
        keep_idx = torch.ops.torchvision.nms(pred_box, pred_scores, iou_threshold)
        keep_boxes = boxes[keep_idx]
        keep_boxes = keep_boxes[keep_boxes[:, -1] >= scores_threshold]
        return keep_boxes

    def preprocess_boxes(self, inputs_ts):
        """
        inputs_ts: (B, S, S, C + 5B) = (B,7,7,30)
        layout: [0:C]=class, [C:C+4B]=boxes, [C+4B:C+5B]=conf
        回傳: pred_boxes (N, 6) -> [class, x1, y1, x2, y2, score]（座標在原圖 pixel 空間）
              batch_ids (N,) -> 對應哪一張圖（避免跨 batch 做 NMS）
        """
        device = inputs_ts.device
        B, S, _, D = inputs_ts.shape
        C = self.class_num
        Bbox = self.numOfBox

        assert D == C + 5 * Bbox, f"expect last dim {C + 5*Bbox}, got {D}"

        # 1. 切出 class / box offset / conf
        pred_cls = inputs_ts[..., :C]                       # (B,S,S,C)
        pred_boxes_offset = inputs_ts[..., C:C+4*Bbox]      # (B,S,S,8)
        pred_boxes_offset = pred_boxes_offset.view(B, S, S, Bbox, 4)  # (B,S,S,B,4)
        pred_xy = pred_boxes_offset[..., :2]
        pred_wh = pred_boxes_offset[..., 2:]                          # wh 預測 sqrt，線性輸出
        pred_boxes_offset = torch.cat([pred_xy, pred_wh], dim=-1)      # wh 為 sqrt 比例
        # conf 與訓練一致，線性輸出（YOLOv1 原設定：預測 IoU）
        pred_conf = inputs_ts[..., C+4*Bbox:C+5*Bbox]       # (B,S,S,B)

        # 2. 轉成 center xy in [0,1]、wh ~ [0,1]（與 loss 共用 decode）
        xywh = tools.decode_bbox_offsets(pred_boxes_offset, grid_size=self.grid)  # (B,S,S,B,4)

        x1y1 = xywh[..., :2] - xywh[..., 2:] / 2.0
        x2y2 = xywh[..., :2] + xywh[..., 2:] / 2.0
        boxes_norm = torch.cat([x1y1, x2y2], dim=-1)        # (B,S,S,B,4)，座標 0~1
        boxes_norm = torch.clamp(boxes_norm, 0.0, 1.0)       # 避免出界的亂框

        # 4. class prob per cell → broadcast 到 box 維度
        cls_prob = F.softmax(pred_cls, dim=-1)              # (B,S,S,C)
        cls_prob = cls_prob.unsqueeze(3).expand(B, S, S, Bbox, C)  # (B,S,S,B,C)

        # 5. 針對「每個類別」計算 score = class_prob * conf，避免只保留單一最大類別
        cls_ids = torch.arange(C, device=device).view(1, 1, 1, 1, C)
        cls_ids = cls_ids.expand(B, S, S, Bbox, C)                   # (B,S,S,B,C)
        scores = cls_prob * pred_conf.unsqueeze(-1)                  # (B,S,S,B,C)

        # 6. 展平所有 box×class（總數 B*S*S*Bbox*C），再縮放到原圖座標
        boxes_norm_flatten = boxes_norm.unsqueeze(4)                 # (B,S,S,B,1,4)
        boxes_norm_flatten = boxes_norm_flatten.expand(B, S, S, Bbox, C, 4)
        boxes_norm_flatten = boxes_norm_flatten.reshape(-1, 4)       # (N,4)
        scores_flatten = scores.reshape(-1, 1)                       # (N,1)
        cls_idx_flatten = cls_ids.reshape(-1, 1).float()             # (N,1)

        # batch id 標籤，避免跨 batch NMS
        batch_ids = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
        batch_ids = batch_ids.expand(B, S, S, Bbox, C).reshape(-1)

        # 7. Scale 和 Concatenate (保持原來的縮放邏輯)
        H, W = self.orig_img_size.to(device)
        scale = torch.tensor([W, H, W, H], device=device)
        boxes_xyxy = boxes_norm_flatten * scale          # (N,4)

        pred_boxes = torch.cat([cls_idx_flatten, boxes_xyxy, scores_flatten], dim=-1) # (N,6)
        return pred_boxes, batch_ids

    def forward(self, inputs_ts):
        pred_boxes, batch_ids = self.preprocess_boxes(inputs_ts)
        B = inputs_ts.size(0)
        results = []

        for b in range(B):
            result_boxes = torch.zeros((0, 6), device=inputs_ts.device)
            batch_mask = (batch_ids == b)
            if not batch_mask.any():
                results.append(result_boxes)
                continue
            pred_boxes_b = pred_boxes[batch_mask]

            for i in range(self.class_num):
                class_mask = (pred_boxes_b[:, 0] == float(i))
                _pred_boxes = pred_boxes_b[class_mask]
                if _pred_boxes.shape[0] == 0:
                    continue
                if self.apply_nms:
                    keep_boxes = self.nms(_pred_boxes, self.iou_threshold, self.scores_threshold)
                else:
                    keep_boxes = _pred_boxes[_pred_boxes[:, -1] >= self.scores_threshold]
                result_boxes = torch.cat([result_boxes, keep_boxes], dim=0)
            results.append(result_boxes)

        if B == 1:
            return results[0]
        return results
