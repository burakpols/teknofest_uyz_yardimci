from ast import arg
from typing import List
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import os
import argparse


class OtoLabel:
    """
    Run otolabel function for labeling pists.
    """

    def __init__(self) -> None:
        pass

    def xyxy2xywhn(self, xyxy: np.ndarray, w: int = 640, h: int = 640) -> np.ndarray:
        """
        Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        Args:
            xyxy (np.ndarray): [xmin,ymin,xmax,ymax] float array.
            w (int, optional): Width of image. Defaults to 640.
            h (int, optional): Height of image. Defaults to 640.
        Returns:
            np.ndarray: normalized xcenter, ycenter, width and height float array.
        """

        xywhn = np.copy(xyxy)
        xywhn[0] = ((xyxy[0] + xyxy[2]) / 2) / w  # x center
        xywhn[1] = ((xyxy[1] + xyxy[3]) / 2) / h  # y center
        xywhn[2] = (xyxy[2] - xyxy[0]) / w  # width
        xywhn[3] = (xyxy[3] - xyxy[1]) / h  # height
        return xywhn

    def xyxy2xywh(self, xyxy: np.ndarray) -> np.ndarray:
        """
        Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right

        Args:
            xyxy (np.ndarray): [xmin,ymin,xmax,ymax] float array.
            w (int, optional): Width of image. Defaults to 640.
            h (int, optional): Height of image. Defaults to 640.

        Returns:
            np.ndarray: xmin, ymin, width and height float array.
        """

        xywh = np.copy(xyxy)
        xywh[0] = int(xyxy[0])  # xmin
        xywh[1] = int(xyxy[1])  # ymin
        xywh[2] = int(xyxy[2] - xyxy[0])  # width
        xywh[3] = int(xyxy[3] - xyxy[1])  # height
        return xywh.astype(np.int16)

    def xyMaxMin(self, contours: tuple) -> List:
        """
        Find xmin, ymin, xmax and ymax from contours. if just one object in frame, return true list.

        Args:
            contours (tuple): contours of object.

        Returns:
            List: xmin,ymin,xmax,ymax list for object bounding box.
        """
        xmax_list = []
        ymax_list = []
        xmin_list = []
        ymin_list = []
        for c in contours:
            xs = c[:, 0, 0]
            ys = c[:, 0, 1]
            xmax_list.append(xs.max())
            ymax_list.append(ys.max())
            xmin_list.append(xs.min())
            ymin_list.append(ys.min())

        xmax = max(xmax_list)
        ymax = max(ymax_list)
        xmin = min(xmin_list)
        ymin = min(ymin_list)

        return [xmin, ymin, xmax, ymax]

    def writer(self, xywh: np.ndarray, class_: str, save: str) -> None:
        """
        Txt writer with YOLO format but coco coordinates like (class xmin ymin w h) 

        Args:
            xywh (np.ndarray): object coordinate based xmin, ymin, width, height.
            class_ (str): object class like uap, uai.
            save (str): save path and its name like (./path/name.txt)
        """
        f = open(save, "a")
        line_ = f"{class_} {str(xywh[0])} {str(xywh[1])} {str(xywh[2])} {str(xywh[3])}\n"
        f.writelines(line_)
        f.close()

    def otoLabel(self, img_path: str, class_: str, save_path: str, img_type: str = "jpg", label_format: str = "coco") -> None:
        """
        Automatic label producer based on giving object features.

        Args:
            img_path (str): Images folder path.
            class_ (str): which class write for object like ("0").
            save_path (str): Txt save folder.
            img_type (str, optional): Image type like (png,jpg). Defaults to "jpg".
            format (str, optional): Annotation format. (yolo or coco)

        Raises:
            Exception: There are only two classes, "0" and "1". Otherwise exception.
        """
        if class_ == "uap":
            lower = np.array([15, 10, 175])
            upper = np.array([62, 85, 255])
        elif class_ == "uai":
            lower = np.array([120, 50, 160])
            upper = np.array([135, 170, 245])

        else:
            raise Exception("Undefined class.")

        if img_path[-1] != "/":
            img_path += "/"
        if save_path[-1] != "/":
            save_path += "/"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        images = glob(img_path+f"*.{img_type}")

        for image in tqdm(images):
            img_name = image.rsplit("\\")[-1].rsplit(".")[0]
            img = cv2.imread(image)
            h, w, c = img.shape
            blur = cv2.GaussianBlur(img, (131, 131), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(blur, blur, mask=mask)
            gray = res[:, :, 0]
            kernelSize = (9, 11)
            opIterations = 3
            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            dilateImage = cv2.morphologyEx(
                gray, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_CONSTANT)

            contours, hierarchy = cv2.findContours(
                dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            xy_list = self.xyMaxMin(contours)
            xyxy = np.array(xy_list, np.float32)
            save = save_path + img_name+".txt"

            if class_ == "uap":
                cls = "0"
            elif class_ == "uai":
                cls = "1"
            else:
                raise Exception("Give exceptable class. uap or uai.")

            if label_format == "yolo":
                xywh = self.xyxy2xywhn(xyxy, w, h)
            elif label_format == "coco":
                xywh = self.xyxy2xywh(xyxy)
            else:
                raise Exception("Give exceptable format. yolo or coco.")

            self.writer(xywh, cls, save)


def parseOpt() -> arg:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str,
                        default='./images', help='image path')
    parser.add_argument('--object-class', type=str, default='0',
                        help='class for save txt')
    parser.add_argument('--save-path', type=str,
                        default='./labels', help='labels path txt')
    parser.add_argument('--image-type', type=str,
                        default='jpg', help='image type')
    parser.add_argument('--label-format', type=str,
                        default='coco', help='annotation format. yolo or coco.')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parseOpt()
    ol = OtoLabel()
    ol.otoLabel(opt.images_path, opt.object_class,
                opt.save_path, opt.image_type,opt.label_format)
