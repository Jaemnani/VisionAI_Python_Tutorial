from openvino.runtime import Core # AI 모델 수행 라이브러리
import numpy as np # 파이썬 수학 라이브러리
import cv2 # 파이썬 영상처리 라이브러리

# 모델 경로
model_xml = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/models/deeplabv3.xml"
# model_bin = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/models/deeplabv3.bin"
model_bin = model_xml.replace(".xml", ".bin")

# 이미지 경로
image_name = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/Datas/bb_image.png"
#image_name = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/Datas/hi_image.png"

# AI 모델 수행 코드
core = Core() # AI 모델 코어
model = core.read_model(model=model_xml, weights=model_bin) # 코어에 저장된 모델 로드
compiled_model = core.compile_model(model=model, device_name="CPU") # 수행 모드로 컴파일

# 이미지 로드
img = cv2.imread(image_name)
resized_img = cv2.resize(img, (513, 513))
input_image = np.expand_dims(resized_img, axis=0)

# AI 모델 수행 및 결과 출력
infer_request = compiled_model.create_infer_request()
input_name = compiled_model.input().any_name
results = infer_request.infer(inputs={input_name: input_image})
output = infer_request.get_output_tensor()
output_data = output.data.squeeze()

# 결과 정리
mask = ((output_data - output_data.min()) * 255.) / (output_data.max() - output_data.min()) # min-max normalization
mask_3ch = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
stacked_image = np.hstack((resized_img, mask_3ch))

# 이미지 창 띄우기
cv2.imshow("original and segmentation", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지로 저장
cv2.imwrite("./mask.jpg", mask)