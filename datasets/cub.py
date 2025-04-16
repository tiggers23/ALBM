import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


base_file_path = "/home/tiggers/mydisk/lql/datasets/cub/cub-base.txt"
novel_file_path = "/home/tiggers/mydisk/lql/datasets/cub/cub-novel.txt"

@DATASET_REGISTRY.register()
class CUB(DatasetBase):

    dataset_dir = "cub/CUB_200_2011"

    def __init__(self, cfg):
        #root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        #self.split_path = os.path.join(self.dataset_dir, "split_ours_CUB.json")
        self.split_path = os.path.join(self.dataset_dir, "split_ours_CUB.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
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
            sort_cls[value] = cls_name[value]#.split('.')[1].lower().replace('_', ' ')
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
            sort_cls_novel[value] = cls_name_novel[value]#.split('.')[1].lower().replace('_', ' ')
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
