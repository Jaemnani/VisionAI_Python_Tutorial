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
        self.input_image = None
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def inference(self, input_image):
        batch_img = self.preprocessing(input_image)
        results = self.infer_request.infer(inputs={self.input_name: batch_img})
        output = self.infer_request.get_output_tensor()
        pred = output.data.argmax(1)[0]
        self.last_output = pred
        return pred
    
    def preprocessing(self, input_image):
        resized_img = cv2.resize(input_image, (513, 513))
        self.input_image = resized_img
        cvt_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        tmp_img = cvt_img.astype(np.float32)
        std_img = tmp_img / 255.
        normalized_img = (std_img - self.mean) / self.std
        tmp_img = normalized_img.transpose(2,0,1)
        batch_img = np.expand_dims(tmp_img, axis=0)
        return batch_img        
    
    def get_mask_image(self):
        if isinstance(self.last_output, np.ndarray):
            mask = ((self.last_output - self.last_output.min()) * 255.) / (self.last_output.max() - self.last_output.min())
        else:
            print("last_output is None.")
            return -1
        return mask
    
    def get_stack_image(self):
        mask = self.get_mask_image()
        mask_3ch = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return np.hstack((self.input_image, mask_3ch))
        
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