import os
import random
from pathlib import Path
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class PascalDataset(Dataset):
    def __init__(self, txt_sample_path, transform=None, S=7, B=2, classes=20, label_dir=None, img_size=448, hflip_prob=0.5):
        # 類別名稱及其對應的 ID
        self.classes_name = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.classes_to_id = {name: idx for idx, name in enumerate(self.classes_name)}

        # 初始化參數
        self.S = S
        self.B = B
        self.classes = classes
        self.transform = transform
        self.txt_sample_path = os.path.expanduser(txt_sample_path)
        self.img_size = img_size
        self.hflip_prob = hflip_prob

        # 讀取 train.txt
        with open(self.txt_sample_path, 'r') as f:
            self.dataset = [line.strip() for line in f.readlines()]

        # 推導標籤所在資料夾（train.txt 同層的 sample/）
        if label_dir is None:
            dataset_root = os.path.dirname(self.txt_sample_path)
            self.label_dir = os.path.join(dataset_root, "sample")
        else:
            self.label_dir = os.path.expanduser(label_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 取得影像路徑
        img_path = self.dataset[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 加載影像
        image = cv.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # 保持 uint8 讓 ToTensor 自動縮放到 [0,1]
        image = cv.resize(image, (self.img_size, self.img_size))

        # 初始化標籤矩陣
        label = np.zeros((self.S, self.S, 5 * self.B + self.classes))

        # 取得標籤文件路徑（優先 label_dir，其次依照影像路徑推斷 VOC 結構）
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        candidates = []
        if self.label_dir:
            candidates.append(Path(self.label_dir) / f"{sample_name}.txt")
        img_path_obj = Path(img_path)
        candidates.append(img_path_obj.parent.parent / "sample" / f"{sample_name}.txt")

        label_path = None
        for cand in candidates:
            if cand.exists():
                label_path = cand
                break
        if label_path is None:
            raise FileNotFoundError(f"Label file not found. Tried: {', '.join(str(c) for c in candidates)}")

        # 讀取標籤並填充到矩陣中
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        do_flip = random.random() < self.hflip_prob
        if do_flip:
            image = cv.flip(image, 1)  # 水平翻轉

        for line in lines:
            class_id, x_center, y_center, width, height = self.split_object(line)

            if do_flip:
                x_center = 1.0 - x_center  # flip 後的中心

            grid_x = int(self.S * x_center)
            grid_y = int(self.S * y_center)
            cell_x = x_center * self.S - grid_x
            cell_y = y_center * self.S - grid_y
            conf_start = self.classes + 4 * self.B  # 28
            
            # YOLOv1 假設一個 cell 只負責一個物體；若已填過則跳過剩餘物體
            if label[grid_y, grid_x, conf_start:conf_start + self.B].sum() > 0:
                continue

            label[grid_y, grid_x, class_id] = 1.0

            sqrt_w = np.sqrt(width)
            sqrt_h = np.sqrt(height)
            # 兩個 box 都填入同一 GT，訓練時讓 IoU 較好的負責，避免第二個 box 閒置
            label[grid_y, grid_x, self.classes:self.classes + 4] = [cell_x, cell_y, sqrt_w, sqrt_h]      # box1
            label[grid_y, grid_x, self.classes + 4:self.classes + 8] = [cell_x, cell_y, sqrt_w, sqrt_h]  # box2
            label[grid_y, grid_x, conf_start]     = 1.0  # conf1 → index 28
            label[grid_y, grid_x, conf_start + 1] = 1.0  # conf2 → index 29

        # 應用圖像轉換（如有）
        if self.transform:
            image = self.transform(image)

        return image, label

    def split_object(self, line):
        """
        將標籤文件中的一行解析為類別 ID 和邊界框信息。
        假設標籤格式為：class_id x_center y_center width height
        """
        values = line.strip().split()
        class_id = int(values[0])
        bbox = list(map(float, values[1:]))
        return [class_id] + bbox

    def create_dataloader(self, batch_size, shuffle=True, num_workers=4, limit=None):
        """
        創建 DataLoader。
        """
        dataset = self
        if limit is not None:
            limit = min(limit, len(self))
            indices = list(range(limit))  # 固定前 limit 筆，便於小集合過擬合測試
            dataset = Subset(self, indices)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
if __name__ == "__main__":
    dataset = PascalDataset(
        txt_sample_path="~/dataset/VOC2012/train.txt",
        transform=None  # 可傳入你的影像轉換
    )

    dataloader = dataset.create_dataloader(batch_size=16, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break
