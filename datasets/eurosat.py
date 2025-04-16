import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}

base_file_path = "/home/STU/lql/datasets/EuroSAT/EuroSAT-base.txt"
novel_file_path = "/home/STU/lql/datasets/EuroSAT/EuroSAT-noveltxt"

@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):

    dataset_dir = "EuroSAT"

    def __init__(self, cfg):
        #root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
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

        cls_name = {}
        for item in train:
            if item.label not in cls_name.keys():
                print("class name:", item.classname)
                print("class label: ", item.label)
                cls_name[item.label] = item.classname

        #sort label as 1, 2, 3 ，，，
        sort_cls = {}
        sort_label = sorted(cls_name.keys())
        for value in sort_label:
            sort_cls[value] = cls_name[value]
            print("sort classname:", sort_cls[value])
            print("sort label:", value)

        if subsample == "base":
            if not os.path.exists(base_file_path):
                with open(base_file_path, 'a') as file:
                    for key, value in sort_cls.items():
                        file.write(f'{value}\n')
            else:
                print("exits base class name")

        cls_name_novel = {}
        for item in test:
            if item.label not in cls_name_novel.keys():
                print("class name:", item.classname)
                print("class label: ", item.label)
                cls_name_novel[item.label] = item.classname

        sort_cls_novel = {}
        sort_label = sorted(cls_name_novel.keys())
        for value in sort_label:
            sort_cls_novel[value] = cls_name_novel[value]
            print("sort classname:", sort_cls_novel[value])
            print("sort label:", value)

        if subsample == "new":
            if not os.path.exists(novel_file_path):
                print("crate novel class name!")
                with open(novel_file_path, 'a') as file:
                    for key, value in sort_cls_novel.items():
                        file.write(f'{value}\n')
            else:
                print("exits novel class name")

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
