import cv2
import os
from tqdm import tqdm


def video2img(video_path: str, result_path: str, fps: int) -> None:

    if result_path[-1]!="/":
        result_path= result_path+"/"
    
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        success, img = cap.read()
        
        if i%fps!=0:
            continue
        
        if success:
            img_name = result_path+ str(i)+ ".png"
            cv2.imwrite(img_name,img)


if __name__=="__main__":
    video2img("./2022_Teknofest_Ulasimda_Yapay_Zeka_Yarismasi_Ornek_Video.mp4","./2022frames/",5)
