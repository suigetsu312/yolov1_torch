import torch    



def grid():
    h = torch.arange(0, 7).view(1, -1).repeat(7, 1).T
    w = torch.arange(0, 7).view(1, -1).repeat(7, 1)
    grid = torch.stack([w, h], dim=-1)
    grid = grid.view(1, 7, 7, 2).float()
    return grid

def calc_iou(boxes1, boxes2):
        b1_xy = boxes1[...,:2]
        b1_wh = boxes1[...,2:]
        b1_wh_half = boxes2[...,2:] /2 
        b1_min = b1_xy - b1_wh_half
        b1_max = b1_xy + b1_wh_half

        b2_xy = boxes2[...,:2]
        b2_wh = boxes2[...,2:]
        b2_wh_half = boxes2[...,2:] /2
        b2_min = b2_xy - b2_wh_half
        b2_max = b2_xy + b2_wh_half

        inter_min = torch.maximum(b1_min,b2_min)
        inter_max = torch.minimum(b1_max,b2_max)
        inter_wh = torch.maximum(inter_max-inter_min, torch.zeros_like(inter_max))

        inter_area = inter_wh[...,0] * inter_wh[...,1]
        b1_area = b1_wh[...,0] * b1_wh[...,1]
        b2_area = b2_wh[...,0] * b2_wh[...,1]
        union_area = b1_area + b2_area - inter_area

        iou = torch.clamp(inter_area/ torch.clamp(union_area, min=1e-6), max=1.0, min = 0.0)
        return iou






