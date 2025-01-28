import cv2
import numpy as np
import random
import os
from tqdm import tqdm


def make_arrow_image(canvas_size = (64, 64), arrow_size = 40, arrow_color = (0, 0, 0), arrow_thickness = 5, arrow_tipLength=3, r_angle = -1):
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype="uint8") * 255
    start_point = (int(canvas_size[0]/2), int(canvas_size[1]/2 + arrow_size/2) )
    end_point   = (int(canvas_size[0]/2), int(canvas_size[1]/2 - arrow_size/2) )
    
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    if r_angle == -1:
        angle = random.randint(0, 360)    
    else:
        angle = r_angle
    
    cv2.arrowedLine(canvas, start_point, end_point, arrow_color, arrow_thickness, tipLength=5)

    # 화살표를 회전시킬 중심점 찾기
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))
    # print(f"Rotated by {angle} degrees")
    
    x = np.arange(rotated.shape[0])
    y = np.arange(rotated.shape[1])
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = distances > np.mean([arrow_size, canvas_size[0], canvas_size[1]])/2
    rotated[mask] = [255, 255, 255]
    
    rotated_sp = np.round([ M[0,0] * start_point[0] + M[0, 1] * start_point[1] + M[0, 2] , 
                   M[1,0] * start_point[0] + M[1, 1] * start_point[1] + M[1, 2] ]).astype(int)
    rotated_ep = np.round([ M[0,0] * end_point[0] + M[0, 1] * end_point[1] + M[0, 2] , 
                   M[1,0] * end_point[0] + M[1, 1] * end_point[1] + M[1, 2] ]).astype(int)
    
    return rotated, angle, rotated_sp, rotated_ep

def main():
    image_path = "./images/"
    direct_path = "./directions/"
    keypoints_path = "./keypoints/"

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(direct_path, exist_ok=True)
    os.makedirs(keypoints_path, exist_ok=True)


    for i in tqdm(range(1800)):
        rotated_img, angle, sp, ep = make_arrow_image(r_angle=i%360)
        
        file_name = "%.6d"%(i)
        
        # Save the Image
        cv2.imwrite(image_path + "img_"+file_name + ".jpg", rotated_img)
        
        # Save the Angle
        with open(direct_path + "ang_"+file_name + ".txt", "w") as f:
            f.write(str(angle))
        
        # Save the keypoints
        with open(keypoints_path + "key_"+file_name + ".txt", "w") as f:
            f.write("%d, %d, %d, %d"%(sp[0], sp[1], ep[0], ep[1]))
            
if __name__ == "__main__":
    main()