from openvino.runtime import Core # AI 모델 수행 라이브러리
import numpy as np # 파이썬 수학 라이브러리
import cv2 # 파이썬 영상처리 라이브러리

class Segmentation:
    def __init__(self, model_xml, device="CPU"):
        self.model_xml = model_xml
        self.model_bin = model_xml.replace(".xml", ".bin")
        
        core = Core()
        model = core.read_model(model=self.model_xml, weights=self.model_bin)
        self.compiled_model = core.compile_model(model=model, device_name=device)
        
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_name = self.compiled_model.input().any_name
        self.last_output = None
        
    def inference(self, input_image):
        results = self.infer_request.infer(inputs={self.input_name: input_image})
        output = self.infer_request.get_output_tensor()
        output_data = output.data.squeeze()
        self.last_output = output_data
        return output_data
    
    def get_mask_image(self):
        if isinstance(self.last_output, np.ndarray):
            mask = ((self.last_output - self.last_output.min()) * 255.) / (self.last_output.max() - self.last_output.min())
        else:
            print("last_output is None.")
            return -1
        return mask
        
# image_names = [ "./Datas/bb_image.png", "./Datas/hi_image.png" ]

# 이미지 경로
# image_name = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/Datas/bb_image.png"
image_name = "/User/yejaemun/SynologyDrive/Develop/class/01.Example/Datas/hi_image.png"

# 이미지 로드
img = cv2.imread(image_name)
resized_img = cv2.resize(img, (513, 513))
input_image = np.expand_dims(resized_img, axis=0)

# AI 모델 수행 및 결과 출력
model_path = "/Users/yejaemun/SynologyDrive/Develop/class/01.Example/models/deeplabv3.xml"
model = Segmentation(model_path)
output_data = model.inference(input_image)
mask = model.get_mask_image()

# 결과 정리
mask_3ch = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
stacked_image = np.hstack((resized_img, mask_3ch))

cv2.imshow("original and segmentation", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지로 저장
cv2.imwrite("./mask.jpg", mask)