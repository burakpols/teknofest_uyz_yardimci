from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
import yaml


def xywhn2xywh(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, wh= width-height bbox
    y = [0, 0, 0, 0]
    y[0] = int(w * (x[0] - x[2] / 2))   # top left x
    y[1] = int(h * (x[1] - x[3] / 2))   # top left y
    y[2] = int(w * (x[0] + x[2] / 2))-y[0]   # w bbox
    y[3] = int(h * (x[1] + x[3] / 2))-y[1]   # h bbox
    return y


def yolo2coco(images_path: str, labels_path: str, save_path: str = "./onurvalpist.json",
              image_type: str = "jpg", yaml_file: str = "./yolo2coco.yaml", convert: bool = False) -> None:

    coco = Coco()
    yml = open(yaml_file, "r").read()
    yml = yaml.safe_load(yml)
    categories = yml["categories"]
    for e, category in enumerate(categories):
        coco.add_category(CocoCategory(id=e, name=category))

    if images_path[-1] != "/":
        images_path += "/"
    if labels_path[-1] != "/":
        labels_path += "/"

    images = glob(images_path+f"*.{image_type}")
    labels = glob(labels_path+"*.txt")

    for i, image in enumerate(tqdm(images)):
        img_name = image.rsplit("\\")[-1]
        width, height = Image.open(image).size
        coco_image = CocoImage(file_name=img_name, height=height, width=width)
        lines = open(labels[i], "r").read().splitlines()
        for line in lines:
            line = line.rsplit(" ")
            if convert:
                bbox = [float(x) for x in line[1:]]
                bbox = xywhn2xywh(bbox, width, height)
                line = [float(line[0]), *bbox]
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[int(line[1]), int(line[2]),
                          int(line[3]), int(line[4])],
                    category_id=int(line[0]),
                    category_name=categories[int(line[0])]
                )
            )

        coco.add_image(coco_image)

    save_json(data=coco.json, save_path=save_path)


def parseOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str,
                        default='./images', help='image path')
    parser.add_argument('--labels-path', type=str, default='./labels',
                        help='labels path')
    parser.add_argument('--save-path', type=str,
                        default='./onurvalpist.json', help='labeled images save path')
    parser.add_argument('--image-type', type=str,
                        default='jpg', help='image type')
    parser.add_argument('--yaml-file', type=str,
                        default='./yolo2coco.yaml', help='yaml file for configs like categories information')
    parser.add_argument('--convert', action='store_true',
                        help='convert from yolo bbox format(normalized xywhn). if false: bbox should be xywh format like coco')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parseOpt()
    yolo2coco(opt.images_path, opt.labels_path, opt.save_path,
              opt.image_type, opt.yaml_file, opt.convert)
