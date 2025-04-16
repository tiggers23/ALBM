import os
import pickle
import random

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


base_file_path = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-base.txt"
novel_file_path = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-novel.txt"

@DATASET_REGISTRY.register()
class DescribableTextures(DatasetBase):

    dataset_dir = "dtd/dtd"

    def __init__(self, cfg):
        #root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        root = cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
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
                print("class name:", item.classname)
                print("class label: ", item.label)
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

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
