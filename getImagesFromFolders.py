from typing import Optional
from glob import glob
from tqdm import tqdm
import shutil
import os


def getImagesFromFolders(folders_path: str, result_path: str, img_type: str = "jpg", name: Optional[str] = None) -> None:
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if result_path[-1] != "/":
        result_path += "/"

    if folders_path[-1] != "/":
        folders_path += "/"

    folders_path = folders_path + f"**/*.{img_type}"
    folders = glob(folders_path, recursive=True)

    for i, img in tqdm(enumerate(folders), desc= "images moving.."):
        if name != None:
            result = result_path + f"{name}_{i}.{img_type}"
        else:
            result = result_path
        shutil.move(img, result)


if __name__ == "__main__":
    getImagesFromFolders("../dataset/teknofest/resimler/2_2",
                         "../dataset/1_3_1", name="131")
