import os
from tqdm import tqdm


def visdrone2yoloxywh(annotations_path: str, save_path: str) -> None:
    if annotations_path[-1] != "/":
        annotations_path += "/"
    if save_path[-1] != "/":
        save_path += "/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for anno in tqdm(os.listdir(annotations_path)):
        label_file = open(save_path + anno, "a")
        txt = annotations_path+anno
        lines = open(txt, "r").read().splitlines()
        new_line_list = []
        for line in lines:
            line_list = line.rsplit(",")
            bbox = line_list[:4]
            category = int(line_list[5])
            if category == 0 or category == 11:
                continue

            new_cat = category-1
            new_line = " ".join(x for x in bbox)
            new_line = f"{new_cat} {new_line}\n"
            new_line_list.append(new_line)

        label_file.writelines(x for x in new_line_list)
        label_file.close()


if __name__ == "__main__":
    visdrone2yoloxywh("../dataset/visdrone/VisDrone2019-DET-val/annotations",
                      "../dataset/visdrone/VisDrone2019-DET-val/labels")
