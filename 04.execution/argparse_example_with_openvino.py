import cv2
import numpy as np # 파이썬 수학 라이브러리
from openvino_classes import Segmentation
import argparse

def main(args):
    input_image = cv2.imread(args.image_path)
    
    model = Segmentation(args.model_path)
    
    output_data = model.inference(input_image)
    
    stacked_image = model.get_stack_image()
    
    if args.show_flag == True:    
        cv2.imshow("Original and Segmentation", stacked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if args.save == True:
        cv2.imwrite('./result.jpg', stacked_image)

def parse_augments():
    parser = argparse.ArgumentParser(description="Demo. how to use Argparse.")
    
    parser.add_argument('-i', '--image_path', 
                    required=True, 
                    help="image_path for input_image.",
                    default="./images/FoodSeg103/train/00000001.jpg")

    parser.add_argument('-m', '--model_path', 
                    required=True, 
                    help="Model path.",
                    default="./models/best_deeplabv3_mobilenet_food_os16.xml")

    parser.add_argument('-s', '--show',
                    required=False,
                    dest="show_flag",
                    help="Result Show or not",
                    action="store_true"
                    )
    
    parser.add_argument("--save",
                    required=False,
                    help="Save result or not",
                    action="store_true"
                    )

    parser.add_argument('-v', '--version',
                    action='version',
                    version="%(prog)s Version 1.0")
    
    return parser

if __name__ == "__main__":
    args = parse_augments().parse_args()
    main(args)