import cv2
import numpy as np # 파이썬 수학 라이브러리
from openvino_classes import Segmentation

        
path = "./images/FoodSeg103/train/00000001.jpg"

# 이미지 로드
input_image = cv2.imread(path)

# AI 모델 수행 및 결과 출력
model_path = "./models/best_deeplabv3_mobilenet_food_os16.xml"
model = Segmentation(model_path)
output_data = model.inference(input_image)

stacked_image = model.get_stack_image()

cv2.imshow("original and segmentation", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지로 저장
cv2.imwrite("./result.jpg", stacked_image)