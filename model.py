import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
import tools
class YOLOv1(torch.nn.Module):
    def __init__(self, batch_size=16, num_classes=20, pretrained=True):
        super(YOLOv1, self).__init__()
        self.backborn = timm.create_model('darknet53', pretrained=pretrained, features_only=True)
        self.backborn.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),  # 強制輸出 7x7
            torch.nn.Flatten(),
            torch.nn.Linear(1024 * 7 * 7, 496),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(496, 1470),
        )

    def forward(self, x):
        out = self.backborn(x)
        return out.view(-1, 7, 7, 30)

class YOLOv1Loss(torch.nn.Module):
    def __init__(s  elf, batch_size=16, 
                 num_classes=20, 
                 lambda_coord=5, 
                 lambda_obj =1, 
                 lambda_noobj=0.5,
                 lambda_class=1, ):
        super(YOLOv1Loss, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def coordinate_loss(self, pred_xywh, true_xywh):
        coord_loss = torch.sum((pred_xywh - true_xywh) ** 2, dim=4)
        return (self.lambda_obj) * coord_loss

    def object_loss(self, iou, pred_conf):
        return self.lambda_coord * torch.sum((pred_conf - iou) ** 2, dim=3, keepdim=True)

    def noobject_loss(self, iou, pred_conf):
        return self.lambda_noobj * torch.sum((pred_conf - iou) ** 2, dim=3, keepdim=True)

    def class_loss(self, pred_classes, true_classes):
        # 將 true_classes 從 one-hot 轉換為類別索引
        true_classes = torch.argmax(true_classes, dim=-1)  # 轉換為 (16, 7, 7)
        # 調整 pred_classes 的形狀為 (batch_size, num_classes, height, width)
        pred_classes = pred_classes.permute(0, 3, 1, 2)  # (16, 7, 7, 20) -> (16, 20, 7, 7)

        class_loss = self.lambda_class * nn.functional.cross_entropy(pred_classes, true_classes, reduction='none')
        return class_loss
    def forward(self, y_pred, y_true):
        
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(-1, 7, 7, 30)

        ypred_bbox_offset = y_pred[..., 20:28].view(-1, 7, 7, 2, 4)
        ytrue_bbox_offset = y_true[..., 20:28].view(-1, 7, 7, 2, 4)

        grid_offsets = tools.grid().expand(batch_size,-1,-1,-1).unsqueeze(-1).to(y_pred.device)
        zzz = ypred_bbox_offset[..., :2]
        ypred_bbox = torch.cat([ypred_bbox_offset[..., :2] + grid_offsets, 
                                ypred_bbox_offset[..., 2:]], dim=-1)
        ytrue_bbox = torch.cat([ytrue_bbox_offset[..., :2] + grid_offsets, 
                                ytrue_bbox_offset[..., 2:]], dim=-1)

        respon_mask = (y_true[..., 28:] > 0).float()

        conf_pred = y_pred[..., 28:].view(-1, 7, 7, 2)
        iou_between_pred_true_box = tools.calc_iou(ypred_bbox, ytrue_bbox).to(y_pred.device)
        # iou_between_pred_true_box: (B, 7, 7, 2)
        best_box = torch.argmax(iou_between_pred_true_box, dim=-1)  # shape: (B, 7, 7)

        # one-hot 轉成 mask: (B, 7, 7, 2)
        object_mask = torch.nn.functional.one_hot(best_box, num_classes=2).float()

        # 把 mask 跟 respon_mask 結合，respon_mask shape: (B, 7, 7, 1)
        object_mask = object_mask * respon_mask  # shape: (B, 7, 7, 2)

        # 反過來是 no-object
        noobject_mask = 1 - object_mask

        coord_loss = torch.sum(self.coordinate_loss(ypred_bbox_offset, ytrue_bbox_offset) * object_mask, dim=-1)
        object_loss = torch.sum(self.object_loss(conf_pred, iou_between_pred_true_box) * object_mask, dim=-1)
        no_object_loss = torch.sum(self.noobject_loss(conf_pred, iou_between_pred_true_box) * noobject_mask, dim=-1)

        pred_class = y_pred[..., :20].view(-1, 7, 7, 20)
        true_class = y_true[..., :20].view(-1, 7, 7, 20)
        class_loss = self.class_loss(pred_class, true_class) * respon_mask[..., 0]

        loss = torch.sum(coord_loss + object_loss + no_object_loss + class_loss, dim=[1, 2]) 
        return loss.mean()
    
class YOLOv1Head(nn.Module):
    def __init__(self, 
                 orig_img_size,
                 class_num=20,
                 iou_threshold=1.0,
                 scores_threshold=0.8,
                 yolo_img_size=(224, 224),
                 name='yolov1_head'):
        super(YOLOv1Head, self).__init__()
        self.iou_threshold = iou_threshold
        self.scores_threshold = scores_threshold
        self.yolo_img_size = torch.tensor(yolo_img_size, dtype=torch.float32)
        self.orig_img_size = torch.tensor(orig_img_size, dtype=torch.float32)
        self.class_num = class_num

    @staticmethod
    def nms(boxes, iou_threshold, scores_threshold):
        """
        Non-Max Suppression for the given boxes.
        Args:
            boxes: Tensor of shape (N, 6) -> [class, x1, y1, x2, y2, score]
            iou_threshold: IoU threshold for suppression
            scores_threshold: Minimum score threshold for valid boxes
        """
        pred_box = boxes[:, 1:5]
        pred_scores = boxes[:, -1]
        keep_idx = torch.ops.torchvision.nms(pred_box, pred_scores, iou_threshold)
        keep_boxes = boxes[keep_idx]
        keep_boxes = keep_boxes[keep_boxes[:, -1] >= scores_threshold]
        return keep_boxes

    def preprocess_boxes(self, inputs_ts):
        """
        Preprocess predictions to convert from YOLO format to box format.
        Args:
            inputs_ts: Tensor of shape (batch, grid_size, grid_size, 5*B + C)
        Returns:
            pred_boxes: Tensor of shape (N, 6) -> [class, x1, y1, x2, y2, score]
        """
        batch_size, grid_size, _, _ = inputs_ts.shape
        feature_hw = torch.tensor([grid_size, grid_size], dtype=torch.float32)

        # Extract predictions
        pred_classes = inputs_ts[..., :self.class_num]
        pred_xywh = inputs_ts[..., self.class_num:self.class_num + 4]
        pred_conf = inputs_ts[..., -1:]

        # Generate grid and normalize
        grid = tools.grid().to(inputs_ts.device)
        pred_xy = (pred_xywh[..., :2] + grid) / feature_hw[0]
        pred_wh = torch.exp(pred_xywh[..., 2:]) / feature_hw[1]

        # Convert to (x1, y1, x2, y2)
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        # Compute scores
        pred_scores = torch.max(pred_classes * pred_conf, dim=-1).values
        pred_class_indices = torch.argmax(pred_classes, dim=-1).float()

        # Combine results
        pred_boxes = torch.cat([
            pred_class_indices.unsqueeze(-1),
            pred_box.view(-1, 4),
            pred_scores.unsqueeze(-1)
        ], dim=-1)

        return pred_boxes

    def forward(self, inputs_ts):
        """
        Forward pass for the YOLOv1 head.
        Args:
            inputs_ts: Tensor of shape (batch, grid_size, grid_size, 5*B + C)
        Returns:
            result_boxes: Tensor of shape (N, 6) -> [class, x1, y1, x2, y2, score]
        """
        pred_boxes = self.preprocess_boxes(inputs_ts)
        result_boxes = torch.zeros((0, 6), device=inputs_ts.device)

        # Apply NMS per class
        for i in range(self.class_num):
            class_mask = (pred_boxes[:, 0] == i)
            _pred_boxes = pred_boxes[class_mask]
            if _pred_boxes.shape[0] == 0:
                continue
            nms_boxes = self.nms(_pred_boxes, self.iou_threshold, self.scores_threshold)
            result_boxes = torch.cat([result_boxes, nms_boxes], dim=0)

        return result_boxes
