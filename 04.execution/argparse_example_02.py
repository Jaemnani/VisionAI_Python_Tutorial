
import argparse

parser = argparse.ArgumentParser(description="Demo. how to use Argparse.")

parser.add_argument('-i', '--input_image', 
                    required=True, 
                    help="Input Image for inference.",
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

parser.add_argument('-v', '--version',
                    action='version',
                    version="%(prog)s Version 1.0")


args = parser.parse_args()

print(args.input_image)
print(args.model_path)
print(args.show_flag)

print(args)