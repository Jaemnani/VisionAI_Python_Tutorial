import cv2
import numpy as np
import random
import os
from tqdm import tqdm

def create_heatmap(h, w, px, py, sigma=2.0):
    xs = np.arange(w)
    ys = np.arange(h)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))
    return heatmap

def create_keypoint_heatmap(sp, ep, img_size=(64, 64), sigma=2.0):
    heatmap_sp = create_heatmap(img_size[1], img_size[0], sp[0], sp[1], sigma=sigma)
    heatmap_ep = create_heatmap(img_size[1], img_size[0], ep[0], ep[1], sigma=sigma)
    heatmap_2ch = np.stack([heatmap_sp, heatmap_ep], axis=0)
    return heatmap_2ch

def make_arrow_image(
        canvas_size = (64, 64), 
        arrow_size_range = (20, 40), 
        arrow_color = (0, 0, 0), 
        arrow_thickness = 5, 
        arrow_tipLength=3, r_angle = -1,
        shift_range=5,
        noise_std=10,
        background_noise=False
        ):
    
    # arrow_size = random.randint(arrow_size_range[0], arrow_size_range(1))
    arrow_size = random.randint(*arrow_size_range)

    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype="uint8") * 255

    # center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    center_x = canvas_size[1]//2 + random.randint(-shift_range, shift_range)
    center_y = canvas_size[0]//2 + random.randint(-shift_range, shift_range)
    center = (center_x, center_y)
    
    start_point = (int(canvas_size[0]/2), int(canvas_size[1]/2 + arrow_size/2) )
    end_point   = (int(canvas_size[0]/2), int(canvas_size[1]/2 - arrow_size/2) )
    
    if r_angle == -1:
        angle = random.randint(0, 359)    
    else:
        angle = r_angle
    
    cv2.arrowedLine(canvas, start_point, end_point, arrow_color, arrow_thickness, tipLength=0.5) # tiplength = 0~1

    # 화살표를 회전시킬 중심점 찾기
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))
    # print(f"Rotated by {angle} degrees")

    if background_noise:
        noise = np.random.normal(0, noise_std, rotated.shape).astype(np.int16)
        rotated = np.clip(rotated + noise, 0, 255).astype(np.uint8)
    
    x = np.arange(rotated.shape[0])
    y = np.arange(rotated.shape[1])
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = distances > np.mean([arrow_size, canvas_size[0], canvas_size[1]])/2
    rotated[mask] = [255, 255, 255]
    
    # rotated_sp = np.round([ M[0,0] * start_point[0] + M[0, 1] * start_point[1] + M[0, 2] , 
    #                M[1,0] * start_point[0] + M[1, 1] * start_point[1] + M[1, 2] ]).astype(int)
    # rotated_ep = np.round([ M[0,0] * end_point[0] + M[0, 1] * end_point[1] + M[0, 2] , 
    #                M[1,0] * end_point[0] + M[1, 1] * end_point[1] + M[1, 2] ]).astype(int)
    start_point_hom = np.array([start_point[0], start_point[1], 1])
    end_point_hom   = np.array([end_point[0], end_point[1], 1])

    rotated_sp = M.dot(start_point_hom)
    rotated_ep = M.dot(end_point_hom)

    rotated_sp = np.round(rotated_sp).astype(int)
    rotated_ep = np.round(rotated_ep).astype(int)

    # tmp = cv2.circle(rotated.copy(), rotated_sp, radius=3, color=(0, 0, 255), thickness=-1)
    # tmp = cv2.circle(tmp, rotated_ep, radius=3, color=(0, 0, 255), thickness=-1)

    kps_2ch = create_keypoint_heatmap(rotated_sp, rotated_ep, img_size=canvas_size, sigma=2.0)

    return rotated, angle, rotated_sp, rotated_ep

def main():
    image_path = "./images/"
    direct_path = "./directions/"
    keypoints_path = "./keypoints/"
    keypoints_dbg_path = "./kp_dbg/"

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(direct_path, exist_ok=True)
    os.makedirs(keypoints_path, exist_ok=True)
    os.makedirs(keypoints_dbg_path, exist_ok=True)


    for i in tqdm(range(18000)):
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

        keypoint_img = cv2.circle(rotated_img.copy(), sp, radius=3, color=(0, 0, 255), thickness=-1)
        keypoint_img = cv2.circle(keypoint_img, ep, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(keypoints_dbg_path + "key_img_"+file_name + ".jpg", keypoint_img)
            
if __name__ == "__main__":
    main()