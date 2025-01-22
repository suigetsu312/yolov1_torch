import torch
import model
import pascal_dataloader
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
# 假設訓練完成後，將模型和優化器狀態保存為檢查點
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


if __name__ == "__main__":
    # 檢查是否有可用的 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用 GPU
    else:
        device = torch.device("cpu")  # 使用 CPU

    EPOCHS = 172
    LEARNING_RATE = 1e-4
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = pascal_dataloader.PascalDataset(
            txt_sample_path="D:\\prog\\od\\data\\train.txt",
            transform=transform  # 可傳入你的影像轉換
        )
    dataloader = dataset.create_dataloader(batch_size=16)
    yolov1_model = model.YOLOv1()
    yolov1_model.to(device)
    yolov1_loss = model.YOLOv1Loss()
    optimizer = optim.Adam(yolov1_model.backborn.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        yolov1_model.train()
        epoch_loss = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # 向前傳播
            predictions = yolov1_model(images)

            # 計算損失
            loss = yolov1_loss(predictions, labels)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
        save_checkpoint(yolov1_model, optimizer, epoch, epoch_loss, 'checkpoint'+str(epoch+1)+'.pth')

    print("Training complete!")
