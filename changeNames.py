from tqdm import tqdm
from glob import glob
import os


def changeNames(source_path: str, new_name: str, type_: str, plus: int = 0) -> None:
    if source_path[-1] != "/":
        source_path += "/"

    files = glob(source_path+"**")

    for f in tqdm(files):
        os.rename(f, source_path+f"{new_name}_{str(plus)}.{type_}")
        plus += 1
        
    
    print(f"**********{plus}***********")


if __name__ == "__main__":
    changeNames("../dataset/pistler", "pist", "jpg", 0)
