from glob import glob
import shutil
import os
def pairing(images_path,labels_path,target_labels_path):
    if not os.path.exists(target_labels_path):
        os.mkdir(target_labels_path) 
        
    images= glob(images_path+"/*.jpg")
    err=[]
    for img in images:
        name= img.rsplit("\\")[-1].rsplit(".")[0]
        try:
            shutil.copy(f"{labels_path}/{name}.txt",f"{target_labels_path}/{name}.txt")
        except:
            err.append(name)
    
    print(f"not found count: {err}")

if __name__=="__main__":
    images_path="C:/Users/Default.DESKTOP-HLSC8QP/Desktop/pistv2/test/images"
    labels_path= "C:/Users/Default.DESKTOP-HLSC8QP/Desktop/pistv2/test/test"
    target_labels_path= "C:/Users/Default.DESKTOP-HLSC8QP/Desktop/pistv2/test/labels"
    pairing(images_path,labels_path,target_labels_path)