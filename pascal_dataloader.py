import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PascalDataset(Dataset):
    def __init__(self, txt_sample_path, transform=None, S=7, B=2, classes=20):
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

        # 讀取 train.txt
        with open(txt_sample_path, 'r') as f:
            self.dataset = [line.strip() for line in f.readlines()]

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
        image = cv.resize(image, (224, 224)).astype(np.float32)

        # 初始化標籤矩陣
        label = np.zeros((self.S, self.S, 5 * self.B + self.classes))

        # 取得標籤文件路徑
        sample_name = img_path.split('\\')[-1].split('.')[0]
        label_path = os.path.join("D:\\prog\\od\\data\\sample", f"{sample_name}.txt")
        label_path = os.path.normpath(label_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # 讀取標籤並填充到矩陣中
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = self.split_object(line)
                grid_x = int(self.S * x_center)
                grid_y = int(self.S * y_center)
                cell_x = x_center * self.S - grid_x
                cell_y = y_center * self.S - grid_y

                label[grid_y, grid_x, class_id] = 1
                label[grid_y, grid_x, self.classes:self.classes + 4] = [cell_x, cell_y, width, height]

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

    def create_dataloader(self, batch_size, shuffle=True, num_workers=4):
        """
        創建 DataLoader。
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
if __name__ == "__main__":
    dataset = PascalDataset(
        txt_sample_path="D:\\prog\\od\\data\\train.txt",
        transform=None  # 可傳入你的影像轉換
    )

    dataloader = dataset.create_dataloader(batch_size=16, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break
