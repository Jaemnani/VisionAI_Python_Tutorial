import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from make_arrow import make_arrow_image, create_keypoint_heatmap, create_heatmap, draw_point_of_keypoints
import copy
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


def find_optimal_batch_size(model, dataset, start_bs=128, device='cuda'):
    bs = start_bs
    while bs > 0:
        try:
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            data_iter = iter(loader)
            imgs, labels = next(data_iter)
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)  # GPU 메모리 사용 시도
            return bs
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                bs //= 2  # 배치 사이즈 절반으로 줄임
                torch.cuda.empty_cache()  # 캐시 비우기
            else:
                raise e
    raise ValueError("Cannot find suitable batch size! Memory too small?")

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        patience: 유효 손실이 개선되지 않아도 기다리는 횟수
        delta: 개선되었다고 여길 최소 변화량
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            # 개선됨
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 개선 안됨
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class ArrowKeypointsDataset(Dataset):
    # def __init__(self, image_dir, angle_dir, transform=None):
    def __init__(self, image_dir, label_dir, target_size=(64, 64), sigma=2.0, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob(self.image_dir + "*.jpg")
        # self.angle_dir = angle_dir
        self.label_dir = label_dir
        self.target_size = target_size
        
        self.sigma = sigma
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_dir + os.path.basename(img_path).replace("img_", "key_").replace('.jpg', '.txt')
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.target_size)
        
        with open(lbl_path, 'r') as af:
            # angle = int(np.array(af.readlines()).flatten()[0]) / 360.
            # angle = float(np.array(af.readlines()).flatten()[0])
            labels = [pts.split(",") for pts in af.readlines()]
            kps = np.array(labels).flatten().astype(float).reshape(-1,2) # sp, ep
            
        # angle_rad = np.deg2rad(angle)
        # label = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)
        sp, ep  = kps
        heatmap = create_keypoint_heatmap(sp, ep, self.target_size, sigma=self.sigma)

        if self.transform:
            img = self.transform(img)
            heatmap = torch.from_numpy(heatmap).float()
        else:
            img = torch.from_numpy(img).unsqueeze(0).float()
            heatmap = torch.from_numpy(heatmap).float()
        return img, kps, heatmap

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True로 메모리 절약 가능

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ArrowKeypointHeatmapNet(nn.Module):
    def __init__(self):
        super(ArrowKeypointHeatmapNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 추가된 컨볼루션 계층
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # 계산된 특성 맵의 크기와 필터 수를 기반으로 첫 번째 완전 연결 계층의 입력 크기 조정
        # 두 번의 풀링을 거치므로, 64 -> 32 -> 16 크기의 특성 맵이 됩니다.
        # self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 특성 맵 크기와 채널 수에 맞춰 조정
        # self.fc2 = nn.Linear(128, 2)

        # self.dropout = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=3, padding=1) # heatmap 2, sp, ep

    def forward(self, x): # torch.Size([32, 1, 64, 64])
        # x = self.pool(self.relu(self.conv1(x))) # torch.Size([32, 16, 32, 32])
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # x = self.pool(self.relu(self.conv2(x))) # torch.Size([32, 32, 16, 16])
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # x = self.pool(self.relu(self.conv3(x))) # torch.Size([32, 64, 8, 8])
        x = self.relu(self.bn3(self.conv3(x)))
        # x = self.pool(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        return x

class DeeperArrowKeypointHeatmapNet(nn.Module):
    def __init__(self):
        super(DeeperArrowKeypointHeatmapNet, self).__init__()
        # 첫 번째 블록: (1 -> 16 -> 16), 풀링
        self.block1 = nn.Sequential(
            ConvBNReLU(1, 16),      # (N,1,H,W) -> (N,16,H,W)
            ConvBNReLU(16, 16),     # (N,16,H,W) -> (N,16,H,W)
            nn.MaxPool2d(2)         # (N,16,H,W) -> (N,16,H/2,W/2)
        )

        # 두 번째 블록: (16 -> 32 -> 32), 풀링
        self.block2 = nn.Sequential(
            ConvBNReLU(16, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2)
        )

        # 세 번째 블록: (32 -> 64 -> 64), 풀링
        self.block3 = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2)
        )

        # 네 번째 블록: (64 -> 128 -> 128), 풀링
        # 입력 이미지 64×64에서 4번 풀링하면 4×4가 됨
        self.block4 = nn.Sequential(
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            # nn.MaxPool2d(2)
        )

        # Fully Connected
        # 최종 feature map 크기는 (128채널, 4×4) → 128*4*4 = 2048
        # self.fc1 = nn.Linear(128 * 4 * 4, 128)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(128, 2)
        self.conv4 = nn.Conv2d(128, 2, kernel_size=3, padding=1) # heatmap 2, sp, ep

    def forward(self, x):
        # 차례로 블록 통과
        x = self.block1(x)  # -> shape: (N,16,32,32)
        x = self.block2(x)  # -> shape: (N,32,16,16)
        x = self.block3(x)  # -> shape: (N,64,8,8)
        x = self.block4(x)  # -> shape: (N,128,4,4)
        
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        x = self.conv4(x)
        # 평탄화
        # x = x.view(x.size(0), -1)  # (N, 128*4*4)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomApply([
#         transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5),
#         # T.ColorJitter(brightness=0.5, contrast=0.5), # 흑백 이미지라면 미미한 효과
#     ], p=0.7),  # 70% 확률로 위 증강들을 적용
#     transforms.Normalize((0.5,), (0.5,))
# ])

dataset = ArrowKeypointsDataset(image_dir='./images/', label_dir='./keypoints/', transform=transform)
train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, drop_last=False)

# model = ArrowKeypointHeatmapNet().to(device)
model = DeeperArrowKeypointHeatmapNet().to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:", total_params)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, setp_size = 30, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

early_stopping = EarlyStopping(patience=10, delta=1e-5, verbose=True)

best_loss = np.inf
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0

num_epochs = 5
for epoch in range(num_epochs):  # 여기서는 10 에포크로 설정
    running_loss = 0.0
    # img, kps, heatmap
    for images, kps, hmaps in train_loader: 

        images = images.to(device)
        # angles = angles.to(device).float().view(-1, 1)  # Ensure angles are the correct shape
        hmaps = hmaps.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, hmaps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]


    model.eval()  # 모델을 평가 모드로 설정
    val_loss = 0.0
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for images, kps, hmaps in valid_loader:
            images = images.to(device)
            # angles = angles.to(device).float().view(-1, 1)
            hmaps = hmaps.to(device)

            outputs = model(images)
            loss = criterion(outputs, hmaps)

            val_loss += loss.item()
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        
    avg_train_loss = (running_loss / len(train_loader))
    avg_valid_loss = (val_loss / len(valid_loader))
    print(f'Epoch [{epoch + 1}/{num_epochs}], LR={current_lr:.6f}, Train Loss: {avg_train_loss:.6f} Valid Loss: {avg_valid_loss:.6f}')
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered !")
        break
model.load_state_dict(best_model_wts)


def img_preprocess(img, torch_trans = transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = torch_trans(img)
    img = img.unsqueeze(0)    
    return img

# def angle_difference(a, b):
#     diff = abs(a - b) % 360
#     if diff > 180:
#         diff = 360 - diff
#     return diff

def post_processing(output):
    # get keypoint from heatmap
    pred_coords = []
    for hm_idx, hm in enumerate(output[0]):
        idx = torch.argmax(hm)
        y = idx // 64
        x = idx % 64
        pred_coords.append((x.item(), y.item()))
    return pred_coords

    return
    # sin_val, cos_val = output[0].cpu().detach().numpy()

    # angle_rad = np.arctan2(sin_val, cos_val)
    # angle_deg = np.rad2deg(angle_rad)
    # if angle_deg < 0:
    #     angle_deg += 360
    # return angle_deg
    # output = output.flatten()[0] * 360.
    # return output.item()

def calc_diff_of_keypoints(gt_kps, pred_kps):
    if len(gt_kps) != len(pred_kps):
        return -1
    
    dist_kps = []
    for gt, pred in zip(gt_kps, pred_kps):
        dist_kp = np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        dist_kps.append(dist_kp)

    return dist_kps

    # dist_sp = np.sqrt((pred_sp[0]-sp[0])**2 + (pred_sp[1]-sp[1])**2)



model.eval()

# angle_threshold = 2.0
# diffs = []
# correct = 0

diff_kps_threshold = 2.0
diffs_kps = []
correct_kps = 0
dst_path = "./kp_pred/"
os.makedirs(dst_path, exist_ok=True)

test_cnt = 100
for i in range(test_cnt):
    test_img, test_angle, test_sp, test_ep = make_arrow_image()
    ground_truth_kps = np.vstack((test_sp, test_ep))
    input_image = img_preprocess(test_img).to(device)
    output = model(input_image)
    result = post_processing(output)

    result_diff = calc_diff_of_keypoints(ground_truth_kps, result)
    result_mean_diff = np.mean(result_diff)

    if result_mean_diff < diff_kps_threshold:
        correct_kps += 1
    diffs_kps.append(result_mean_diff)
    print("gt->pred : sp(%.2f, %.2f)->(%.2f, %.2f)(%.2f), ep(%.2f, %.2f)->(%.2f, %.2f)(%.2f)"%
          (ground_truth_kps[0][0], ground_truth_kps[0][1], result[0][0], result[0][1], result_diff[0], 
           ground_truth_kps[1][0], ground_truth_kps[1][1], result[1][0], result[1][1], result_diff[1]))
    result_img = draw_point_of_keypoints(test_img, result)
    save_path = dst_path + "%.6d.jpg"%(i)
    cv2.imwrite(save_path, result_img)

    # diff = abs(test_angle - result)
    # diff = angle_difference(test_angle, result)
    # if diff < angle_threshold:
    #     correct += 1
    # diffs.append(diff)
    # print("gt:", test_angle, "pred:", result, "diff:", diff)
# diffs = np.array(diffs)
# print("acc: ", float(correct) / test_cnt )
# print("mean/max/min diff:", np.mean(diffs), ",",np.max(diffs),",", np.min(diffs))

print("acc: ", float(correct_kps) / test_cnt )
print("mean/max/min diff:", np.mean(diffs_kps), ",",np.max(diffs_kps),",", np.min(diffs_kps))

print("done")