import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}

base_file_path = "/home/STU/lql/datasets/caltech-101/caltech-101-base.txt"
novel_file_path = "/home/STU/lql/datasets/caltech-101/caltech-101-novel.txt"


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        #root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        cls_name = []
        for item in train:
            if item.classname not in cls_name:
                #print("class name:", item.classname)
                cls_name.append(item.classname)
        if subsample == "base":
            if not os.path.exists(base_file_path):
                with open(base_file_path, 'a') as file:
                    for item in cls_name:
                        if item == "face":
                            file.write("faces\n")
                        elif item == "leopard":
                            file.write("leopards\n")
                        elif item == "motorbike":
                            file.write("motorbikes\n")
                        elif item == "airplane":
                            file.write("airplanes\n")
                        else:
                            file.write(f'{item}\n')
            else:
                print("exits base class name")

        cls_name_novel = []
        for item in test:
            if item.classname not in cls_name_novel:
                #print("class name:", item.classname)
                cls_name_novel.append(item.classname)
            if subsample == "new":
                if not os.path.exists(novel_file_path):
                    print("crate novel class name!")
                    with open(novel_file_path, 'a') as file:
                        for item in cls_name:
                            file.write(f'{item}\n')
                else:
                    print("exits novel class name")

        super().__init__(train_x=train, val=val, test=test)
