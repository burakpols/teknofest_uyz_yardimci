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
        self.clahe = cv2.createCLAHE(clipLimit = 5)

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

    def writer(self, xywhn: np.ndarray, class_: str, save: str) -> None:
        """
        Txt writer with YOLO format like (class xcenter ycenter w h) 

        Args:
            xywhn (np.ndarray): normalized object coordinate based xcenter, ycenter, width, height.
            class_ (str): object class like uap, uai.
            save (str): save path and its name like (./path/name.txt)
        """
        f = open(save, "a")
        line_ = f"{class_} {str(xywhn[0])} {str(xywhn[1])} {str(xywhn[2])} {str(xywhn[3])}\n"
        f.writelines(line_)
        f.close()

    def otoLabel(self, img_path: str, class_: str, save_path: str, img_type: str = "jpg") -> None:
        """
        Automatic label producer based on giving object features.

        Args:
            img_path (str): Images folder path.
            class_ (str): which class write for object like ("0").
            save_path (str): Txt save folder.
            img_type (str, optional): Image type like (png,jpg). Defaults to "jpg".

        Raises:
            Exception: There are only two classes, "0" and "1". Otherwise exception.
        """
        if class_ == "0":
            lower= np.array([81,31,235])
            upper= np.array([102,51,255])
        elif class_ == "1":
            lower= np.array([155,102,215])
            upper= np.array([194,142,255])

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
            img= cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
            blur = cv2.GaussianBlur(img, (55, 55), 0)
            l,a,b= cv2.split(blur)
            cl= self.clahe.apply(l)
            img= cv2.merge((cl,a,b))
            h, w, ch = img.shape
            

            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            hsv= cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(img, img, mask=mask)

            gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

            kernelSize = (11, 11)
            opIterations = 5
            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            dilateImage = cv2.morphologyEx(
                gray, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

            contours, hierarchy = cv2.findContours(
                dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            xy_list = self.xyMaxMin(contours)
            xyxy = np.array(xy_list, np.float32)
            xywhn = self.xyxy2xywhn(xyxy, w, h)
            save = save_path + img_name+".txt"
            self.writer(xywhn, class_, save)


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
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parseOpt()
    ol = OtoLabel()
    ol.otoLabel(opt.images_path, opt.object_class, opt.save_path, opt.image_type)

