# path = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_clobot_pod/annotations/person_keypoints_default.json"
# path = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_techno_pal/annotations/person_keypoints_default.json"

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from make_arrow import color_map
from tqdm import tqdm

def parse_coco_keypoints(ann_file, img_dir, cat_name='person'):
    coco = COCO(ann_file)

    # 2) 원하는 카테고리 ID 가져오기 (예: 사람)
    cat_ids = coco.getCatIds(catNms=[cat_name])
    
    # 3) 해당 카테고리 이미지 ID 가져오기
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    # 4) 이미지 정보 가져오기
    for img_id in img_ids:
        # 4-1) 이미지 메타 데이터 로드
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        height, width = img_info['height'], img_info['width']
        
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            # 이미지 파일이 없으면 스킵
            continue
        
        # 4-2) 해당 이미지의 person(또는 cat_name) 어노테이션 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        annos = coco.loadAnns(ann_ids)

        # 각 객체의 키포인트, bbox 등
        keypoints_list = []
        bbox_list = []
        for ann in annos:
            if 'keypoints' not in ann:
                # keypoints 없는 객체 스킵
                continue
            # 키포인트: [x1, y1, v1, x2, y2, v2, ...] 길이=3* num_keypoints
            kpts = ann['keypoints']
            # COCO에서 visibility: v=0(불확실), v=1(가려짐), v=2(정확히 보임)
            
            keypoints_list.append(kpts)
            bbox_list.append(ann['bbox'])  # [x, y, w, h]
            
        yield {
            'image_id': img_id,
            'file_name': file_name,
            'img_path': img_path,
            'width': width,
            'height': height,
            'keypoints_list': keypoints_list,
            'bbox_list': bbox_list
        }

# 사용 예시
if __name__ == "__main__":

    dbg_path = "./cocokpts_label_dbg/"
    img_save_path = "./images/"
    kpt_save_path = "./keypoints/"
    inzapp_path = "./inzapp/"
    os.makedirs(dbg_path, exist_ok=True)
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(kpt_save_path, exist_ok=True)
    os.makedirs(inzapp_path, exist_ok=True)

    # ann_file = 'annotations/person_keypoints_train2017.json'
    # img_dir  = 'train2017/'  # 실제 경로로 수정

    # ann_file = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_clobot_pod/annotations/person_keypoints_default.json"
    # img_dir = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_clobot_pod/images/default"
    # cat_name = "pod"
    
    ann_file = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_techno_pal/annotations/person_keypoints_default.json"
    img_dir = "/Users/jaemoonye/Downloads/dataset/keypoints/test_keypoints_techno_pal/images/default"
    cat_name = "pallet"
    

    # for data in parse_coco_keypoints(ann_file, img_dir, cat_name='person'):
    for data in parse_coco_keypoints(ann_file, img_dir, cat_name=cat_name):
        img_path = data['img_path']
        kpts_list = data['keypoints_list']  # 여러 person 객체가 있을 수 있음
        bbox_list = data['bbox_list']

        print(f"Image ID: {data['image_id']}, path: {img_path}")
        print(f"Number of {cat_name}: {len(kpts_list)}")
        
        # 예) 첫 번째 사람의 keypoints 출력
        # if len(kpts_list) > 0:
        #     print(f"Keypoints (1st {cat_name}):", kpts_list[0])
        #     print(f"Bbox       (1st {cat_name}):", bbox_list[0])

        img = cv2.imread(img_path)
        img_size = img.shape[::-1][1:]

        ori_img = img.copy()
        list_length = len(kpts_list)
        print(f"{cat_name}'s Kpts length {list_length}")
        for idx in range(list_length):
            bbox = bbox_list[idx]
            bbox_int = np.round(bbox).astype(int)
            x,y,w,h = bbox_int
            wb = int(bbox_int[2] / 10)
            hb = int(bbox_int[3] / 10)

            x = bbox_int[0]-wb if bbox_int[0]>=wb else 0
            y = bbox_int[1]-hb if bbox_int[1]>=hb else 0
            w = bbox_int[2]+hb if bbox_int[2]<img_size[0]-1 else img_size[0]-1
            h = bbox_int[3]+wb if bbox_int[3]<img_size[1]-1 else img_size[0]-1
                
            crop_bbox = np.array([x, y, w, h])
            
            # x, y, w, h = crop_bbox
            crop_img = img.copy()[crop_bbox[1] : crop_bbox[1] + crop_bbox[3], crop_bbox[0] : crop_bbox[0] + crop_bbox[2]]
            crop_ori_img = crop_img.copy()
            kpts = np.array(kpts_list[0]).reshape(-1,3)
            kpts = kpts - np.array([x, y, 0])

            inzapp_label = kpts.copy()
            cimg_size = crop_img.shape[::-1][1:]
            conf = ((inzapp_label[:,-1] == 2) * 1).reshape(-1,1)
            kps = inzapp_label[:,:-1]
            kps[:, 0] /= cimg_size[0]
            kps[:, 1] /= cimg_size[1]
            kps = np.clip(kps, 0, 1)
            inzapp_label = np.hstack((conf, kps))
            

            kpts[:,:2][kpts[:,2]==1] = [-1, -1]
            kpts = kpts[:, :2]
            kpts = np.round(kpts).astype(int)

            for kp_idx, kp in enumerate(kpts):
                if kp[0] < 0 or kp[1] < 0 or kp[0] > img_size[0] or kp[1] > img_size[1]:
                    continue
                # kp = np.round(kp).astype(int)
                crop_img = cv2.circle(crop_img, kp, radius=3, color=color_map[kp_idx], thickness=-1)
            break

            print("")

        INZAPP=True
        if INZAPP:
            save_file_name = os.path.basename(img_path).replace(".jpg", "_DBG.jpg").replace(".png", "_DBG.png")
            save_label_name = os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
            cv2.imwrite(dbg_path + save_file_name, crop_img)
            cv2.imwrite(inzapp_path + os.path.basename(img_path).replace(".png", ".jpg"), crop_ori_img)
            # np.savetxt(inzapp_path+save_label_name, inzapp_label)
            with open(inzapp_path + save_label_name, 'w') as f:
                for i in inzapp_label:
                    line = "%.1f %.6f %.6f\n"%(i[0], i[1], i[2])
                    f.write(line)
                # line = ','.join(str(x) for x in kpts.flatten())
                # f.write(line)
        else:
            save_file_name = os.path.basename(img_path).replace(".jpg", "_DBG.jpg").replace(".png", "_DBG.png")
            save_label_name = os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt")
            cv2.imwrite(dbg_path + save_file_name, crop_img)
            cv2.imwrite(img_save_path + os.path.basename(img_path).replace(".png", ".jpg"), crop_ori_img)
            
            with open(kpt_save_path + save_label_name, 'w') as f:
                line = ','.join(str(x) for x in kpts.flatten())
                f.write(line)
        

        # 이미지를 로드해서 키포인트 시각화하려면:
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # ... (draw keypoints)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        
        # break  # 테스트용, 첫 이미지만
