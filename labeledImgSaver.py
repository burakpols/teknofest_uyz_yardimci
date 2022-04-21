import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import argparse


class LabeledImgSaver:
    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def autoColorChooser(self, class_: str) -> tuple:
        RGB_colors = [(191, 48, 115), (33, 1, 64),
                      (143, 178, 191), (242, 191, 145), (217, 82, 4)]
        if int(class_) == 0:
            ix = 4
        elif int(class_) < len(RGB_colors):
            ix = len(RGB_colors) % int(class_)
        else:
            ix = int(class_) % len(RGB_colors)

        return RGB_colors[ix]

    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0) -> np.ndarray:
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
        y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
        y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
        y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y

        return y.astype(np.int16)

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[0] = x[0]  # xmin
        y[1] = x[1]  # ymin
        y[2] = x[0]+x[2]  # xmax
        y[3] = x[1]+x[3]  # ymax

        return y.astype(np.int16)

    def labeledImgSaver(self, img_path: str, label_path: str, save_path: str = "./labeledImg",
                        img_type: str = "jpg", label_format: str = "yolo") -> None:
        if img_path[-1] != "/":
            img_path += "/"
        if label_path[-1] != "/":
            label_path += "/"
        if save_path[-1] != "/":
            save_path += "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        images = glob(img_path+f"*.{img_type}")
        labels = glob(label_path+"*.txt")

        for image in tqdm(images):
            img_name = image.rsplit("\\")[-1].rsplit(".")[0]
            img = cv2.imread(image)
            H, W, C = img.shape
            for label in labels:
                label_name = label.rsplit("\\")[-1].rsplit(".")[0]

                if img_name == label_name:
                    txt = open(label, "r").read()
                    lines = txt.splitlines()
                    for line in lines:
                        c, x, y, w, h = line.rsplit(" ")
                        color = self.autoColorChooser(c)

                        if c == "0":
                            c = "uap"
                        elif c == "1":
                            c = "uai"

                        if label_format == "yolo":
                            xywh = np.array(
                                [float(x), float(y), float(w), float(h)], np.float32)
                            xyxy = self.xywhn2xyxy(xywh, W, H)
                        elif label_format == "coco":
                            xywh = np.array(
                                [int(x), int(y), int(w), int(h)], np.int16)
                            xyxy = self.xywh2xyxy(xywh)
                        else:
                            raise Exception("Unsupported label format.")

                        img = cv2.rectangle(
                            img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                        img = cv2.putText(
                            img, c, (xyxy[0], xyxy[1]-10), self.font, 1, color, 2, cv2.LINE_AA)

                    cv2.imwrite(save_path+img_name+".jpg", img)
                else:
                    continue


def parseOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str,
                        default='./images', help='image path')
    parser.add_argument('--labels-path', type=str, default='./labels',
                        help='labels path')
    parser.add_argument('--save-path', type=str,
                        default='./labeledImg', help='labeled images save path')
    parser.add_argument('--image-type', type=str,
                        default='jpg', help='image type')
    parser.add_argument('--label-format', type=str,
                        default='yolo', help='label format. yolo or coco.')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parseOpt()
    LIS = LabeledImgSaver()
    LIS.labeledImgSaver(opt.images_path, opt.labels_path,
                        opt.save_path, opt.image_type, opt.label_format)
