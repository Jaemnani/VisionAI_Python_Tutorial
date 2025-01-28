import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from make_arrow import make_arrow_image

class ArrowDataset(Dataset):
    def __init__(self, image_dir, angle_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob(self.image_dir + "*.jpg")
        self.angle_dir = angle_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ang_path = self.angle_dir + os.path.basename(img_path).replace("img_", "ang_").replace('.jpg', '.txt')
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        
        with open(ang_path, 'r') as af:
            angle = int(np.array(af.readlines()).flatten()[0]) / 360.

        if self.transform:
            img = self.transform(img)

        return img, angle
    
class ArrowNet(nn.Module):
    def __init__(self):
        super(ArrowNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 추가된 컨볼루션 계층
        self.pool = nn.MaxPool2d(2, 2)
        # 계산된 특성 맵의 크기와 필터 수를 기반으로 첫 번째 완전 연결 계층의 입력 크기 조정
        # 두 번의 풀링을 거치므로, 64 -> 32 -> 16 크기의 특성 맵이 됩니다.
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 특성 맵 크기와 채널 수에 맞춰 조정
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x): # torch.Size([32, 1, 64, 64])
        x = self.pool(self.relu(self.conv1(x))) # torch.Size([32, 16, 32, 32])
        x = self.pool(self.relu(self.conv2(x))) # torch.Size([32, 32, 16, 16])
        x = self.pool(self.relu(self.conv3(x))) # torch.Size([32, 64, 8, 8])
        x = x.view(-1, 64 * 8 * 8)  # 배치 차원을 유지하면서 나머지 차원을 평탄화
        x = self.relu(self.fc1(x)) # torch.Size([32, 128])
        x = self.fc2(x) # torch.Size([32, 1])
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = ArrowDataset(image_dir='./images/', angle_dir='./directions/', transform=transform)
train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArrowNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# for epoch in tqdm(range(100)):  # 여기서는 10 에포크로 설정
for epoch in range(100):  # 여기서는 10 에포크로 설정
    running_loss = 0.0
    for images, angles in train_loader:
        images = images.to(device)
        angles = angles.to(device).float().view(-1, 1)  # Ensure angles are the correct shape

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, angles)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    model.eval()  # 모델을 평가 모드로 설정
    val_loss = 0.0
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for images, angles in valid_loader:
            images = images.to(device)
            angles = angles.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, angles)

            val_loss += loss.item()
            
    print(f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)} Valid Loss: {val_loss / len(valid_loader)}')

def img_preprocess(img, torch_trans = transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = torch_trans(img)
    return img

def post_processing(output):
    output = output.flatten()[0] * 360.
    return output.item()

def angle_difference(a, b):
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

model.eval()
angle_threshold = 2.5
diffs = []
correct = 0
test_cnt = 100
for i in range(test_cnt):
    test_img, test_angle, test_sp, test_ep = make_arrow_image()
    input_image = img_preprocess(test_img).to(device)
    output = model(input_image)
    result = post_processing(output)
    # diff = abs(test_angle - result)
    diff = angle_difference(test_angle, result)
    if diff < angle_threshold:
        correct += 1
    diffs.append(diff)
    print("gt:", test_angle, "pred:", result, "diff:", abs(test_angle - result))
diffs = np.array(diffs)
print("acc: ", float(correct) / test_cnt )
print("mean/max/min diff:", np.mean(diffs), ",",np.max(diffs),",", np.min(diffs))





print("done")