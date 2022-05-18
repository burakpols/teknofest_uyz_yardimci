from glob import glob
import yaml
import os
from tqdm import tqdm


def labelConverter(converter_yaml: str, data_yaml: str) -> None:
    """convert classes from yolo label files (txt). 

    Args:
        converter_yaml (str): This is main yaml folder for converting settings.
            *   datasets: which folders convert in dataset. ['train', 'valid'].
            *   names: Class names like yolo data.yaml. ['a','b'].
            *   class_num: Determine new class numbers for each classes in the correct order respect to the names.['0','0'] for ['a','b']
            *   target_folder: Save folder path for new labels.

        data_yaml (str): This is data yaml file for information about dataset.
            *   train: Train images path. 
            *   valid: Valid images path. These paths calling according to converter_yaml datasets.
            *   nc: Number of classes.
            *   names: Classes names.


    Raises:
        Exception: Converter yaml names and class_num must have same length.
        Assertion: If converter yaml datasets has a dataset type and it is not in the data yaml, there is exception or 
                   giving path exactly wrong.
    """
    const_yaml = open(converter_yaml, "r").read()
    const_yaml = yaml.safe_load(const_yaml)

    d_yaml = open(data_yaml, "r").read()
    d_yaml = yaml.safe_load(d_yaml)

    classes = {}
    for e, name in enumerate(tqdm(d_yaml["names"], "Analyze classes...")):
        if name.lower() in const_yaml["names"]:
            try:
                classes[f"{e}"] = str(const_yaml["class_num"][const_yaml["names"].index(
                    name.lower())])
            except:
                raise Exception(
                    "Converter yaml file has class_num which is not enought length.")

    for dataset in tqdm(const_yaml["datasets"], "Converting dataset classes..."):

        if not os.path.exists(const_yaml["target_folder"]+"/"+dataset):
            os.makedirs(const_yaml["target_folder"]+"/"+dataset)

        path = d_yaml[dataset].rsplit("/")
        path[-1] = "labels"
        path = "/".join(path)

        labels = glob(path+"/*.txt")

        assert not len(
            labels) == 0, "Cannot find labels. Maybe there is wrong images path in data.yaml"

        for txt in labels:
            f = open(txt, "r").read().splitlines()
            newf = open(const_yaml["target_folder"] + "/" + dataset +
                        "/"+txt.rsplit("\\")[-1], "w")
            new_lines_list = []

            for line in f:
                if line[0] in classes.keys():
                    new_line = line.replace(line[0], classes[line[0]]) + "\n"
                    new_lines_list.append(new_line)

            newf.writelines(new_lines_list)
            newf.close()


if __name__ == "__main__":
    converter_yaml = "A:/teknofest/ulasim/yardimci_kodlar/converter.yaml"
    data_yaml = "A:/teknofest/ulasim/dataset/pistdataset/pistler/sss.v2-pistlerv2.yolov5pytorch/data.yaml"
    labelConverter(converter_yaml, data_yaml)
