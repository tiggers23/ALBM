import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


from tqdm import tqdm
from dassl.evaluation import build_evaluator

import os
import numpy as np


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CUB": 'a photo of a {}, a type of bird',
    'cifar100': 'a photo of a {}',
    'Places365': 'a photo of a {}',
    'cifar10': 'a photo of a {}'
}


#base_file = "/home/tiggers/mydisk/lql/datasets/Place365/places365_standard/places-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/Place365/places365_standard/places-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/cub/cub-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/cub/cub-novel.txt"

base_file = "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-base.txt"
novel_file = "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar100-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar100-novel.txt"

#base_file = "/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10-base.txt"
#novel_file = "/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10-novel.txt"

BASE_FLIE_PATH = {
    'Food101': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-base.txt',
    'Food101_tensor': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-base.txt',
    'OxfordFlowers': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-base.txt",
    'OxfordFlowers_tensor': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-base.txt",
    'OxfordPets': "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-base.txt",
    'OxfordPets_tensor': "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-base.txt",
    'FGVCAircraft': "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-base.txt",
    'FGVCAircraft_tensor': "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-base.txt",
    'DescribableTextures': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-base.txt",
    'DescribableTextures_tensor': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-base.txt",
    'CUB': "/home/tiggers/mydisk/lql/datasets/cub/cub-base.txt",
    'CUB_tensor': "/home/tiggers/mydisk/lql/datasets/cub/cub-base.txt",
}

NOVEL_FLIE_PATH = {
    'Food101': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-novel.txt',
    'Food101_tensor': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-novel.txt',
    'OxfordFlowers': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-novel.txt",
    'OxfordFlowers_tensor': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-novel.txt",
    'OxfordPets': "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-novel.txt",
    'OxfordPets_tensor': "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-novel.txt",
    'FGVCAircraft': "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-novel.txt",
    'FGVCAircraft_tensor': "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-novel.txt",
    'DescribableTextures': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-novel.txt",
    'DescribableTextures_tensor': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-novel.txt",
    'CUB': "/home/tiggers/mydisk/lql/datasets/cub/cub-novel.txt",
    'CUB_tensor': "/home/tiggers/mydisk/lql/datasets/cub/cub-novel.txt",
}

GPT_TEXT_PRE_PATH_LABO = {
    'Food101': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food_labo_classname.pt',
    'OxfordFlowers': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_classname.pt",
    'FGVCAircraft': "/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_classname.pt",
    'DescribableTextures': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_labo_classname.pt",
    'CUB': "/home/tiggers/mydisk/lql/datasets/cub/cub_labo_classname.pt",
}

GPT_TEXT_AFTER_PATH_LABO = {
    'Food101': '/home/tiggers/mydisk/lql/datasets/food101/food-101/food_labo_label.pt',
    'OxfordFlowers': "/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_label.pt",
    'FGVCAircraft': "/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_label.pt",
    'DescribableTextures': "/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_labo_label.pt",
    'CUB': "/home/tiggers/mydisk/lql/datasets/cub/cub_labo_label.pt",
}
#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/Place365/places_iclr23_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/Place365/places_iclr23_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/Place365/places_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/Place365/places_ours_classname.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_classname.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_cupl_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_cupl_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_iclr23_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_iclr23_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_enhance_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_enhance_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_enhance_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_enhance_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_classname.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_cupl_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_cupl_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_enhance_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_enhance_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_enhance_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_enhance_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_ours_classname.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_iclr23_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_iclr23_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_enhance_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_enhance_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_enhance_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_enhance_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_cupl_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_cupl_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_cupl_base_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_cupl_base_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_classname.pt', map_location='cpu', encoding='latin1')




#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_handcraft_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_handcraft_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_iclr23_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_iclr23_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_enhance_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_enhance_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_enhance_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_enhance_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_2_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_2_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_3_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_ours_3_classname.pt', map_location='cpu', encoding='latin1')





#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_enhance_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_enhance_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_enhance_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_enhance_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_classname.pt', map_location='cpu', encoding='latin1')

gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_label.pt', map_location='cpu', encoding='latin1')
gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_label_2_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_classname_2_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_label_2.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_classname_2.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_label_2.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_ours_classname_2.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_cuplbase_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_cuplbase_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_cupl_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_cupl_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_iclr23_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_iclr23_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food101_nocls_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food101_nocls_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food101_nocls_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food101_nocls_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_description_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_description_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_description_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_description_classname.pt', map_location='cpu', encoding='latin1')





#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_cuplbase_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_cuplbase_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/CIFAR10_ours_classname.pt', map_location='cpu', encoding='latin1')



#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_cuplbase_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_cuplbase_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_labo_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_labo_classname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_label_addname.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_classname_addname.pt', map_location='cpu', encoding='latin1')

#gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_ours_label.pt', map_location='cpu', encoding='latin1')
#gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/CIFAR100_ours_classname.pt', map_location='cpu', encoding='latin1')




@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.evaluator_prompt = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.score_existing_data = []
        self.feature_existing_data = []
        self.label_feature = []
        self.label_score = []

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        
        
        if cfg.DATASET.SUBSAMPLE_CLASSES == "base":
            self.base_or_novel = "base"
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "new":
            self.base_or_novel = "novel"
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "all":
            self.base_or_novel = "all"
        
        #################################################################################
        text_feature_img_base = []
        with open(base_file, 'r') as file:
            base_class_name = [line.strip() for line in file.readlines()]

        for cls_name in base_class_name:
            #gpt_text_pre = torch.load(GPT_TEXT_PRE_PATH_LABO[cfg.DATASET.NAME], map_location='cpu', encoding='latin1')
            gpt_sentence = gpt_text_pre[cls_name.lower().replace('_', ' ')]
            #gpt_sentence = gpt_text_pre[cls_name.lower()]
            #gpt_sentence = gpt_text_pre[cls_name.lower().replace('-', ' ')]
            #print(gpt_sentence)
            gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])
            '''
            tokens = [
                clip.tokenize(s) if s else torch.zeros(1, 77, dtype=torch.long) 
                for s in gpt_sentence
            ]

            # 检查 tokens 是否为空，如果为空，则返回一个默认的张量
            if not tokens:
                gpt_sentence = torch.zeros(1, 77, dtype=torch.long)
            else:
                gpt_sentence = torch.cat(tokens)
            

            '''    #print("shape of gpt_sentence: ", gpt_sentence.shape)
            gpt_sentence = gpt_sentence.to('cuda')
            #self.model = self.model.to('cuda')
            with torch.no_grad():
                text_feature = clip_model.encode_text(gpt_sentence)
                    #print("shape of gpt_sentence feature: ", text_feature.shape)
                ''''''
                ###################################################################
                #text_feature = torch.sum(text_feature, dim=0)
                #text_feature = text_feature / text_feature.size(0)
                #text_feature = torch.mean(text_feature, dim=0)
                ###################################################################
                
                text_feature_img_base.append(text_feature)
                #print(text_feature.shape)
                
            #self.model.to('cpu')
        ###########normal
        text_feature_img_base = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_img_base], dim=0)
        print('text feature img base: ', text_feature_img_base.shape)
        text_feature_img_base = text_feature_img_base / text_feature_img_base.norm(dim=-1, keepdim=True)
        text_feature_avg_base = torch.sum(text_feature_img_base, dim=1)
        text_feature_avg_base = text_feature_avg_base / text_feature_img_base.size(1)
        print('text feature avg base: ', text_feature_avg_base.shape)
        #self.text_feature_avg_base = text_feature_avg_base.half()
        '''
        #################description
        text_feature_avg_base = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_img_base], dim=0)
        print('text feature img base: ', text_feature_avg_base.shape)
        '''
        
        text_feature_img_novel = []
        with open(novel_file, 'r') as file:
            novel_class_name = [line.strip() for line in file.readlines()]

        for cls_name in novel_class_name:
            #gpt_text_pre = torch.load(GPT_TEXT_PRE_PATH_LABO[cfg.DATASET.NAME], map_location='cpu', encoding='latin1')
            gpt_sentence = gpt_text_pre[cls_name.lower().replace('_', ' ')]
            #gpt_sentence = gpt_text_pre[cls_name.lower()]
            #gpt_sentence = gpt_text_pre[cls_name.lower().replace('-', ' ')]
            
            gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])
                #print("shape of gpt_sentence: ", gpt_sentence.shape)
            '''
            tokens = [
                clip.tokenize(s) if s else torch.zeros(1, 77, dtype=torch.long) 
                for s in gpt_sentence
            ]
            if not tokens:
                gpt_sentence = torch.zeros(1, 77, dtype=torch.long)
            else:
                gpt_sentence = torch.cat(tokens)
            
            '''
            gpt_sentence = gpt_sentence.to('cuda')
            #self.model = self.model.to('cuda')
            with torch.no_grad():
                text_feature = clip_model.encode_text(gpt_sentence)
                #print("shape of gpt_sentence feature: ", text_feature.shape)
                '''
                ###################################################################
                #text_feature = torch.sum(text_feature, dim=0)
                #text_feature = text_feature / text_feature.size(0)
                text_feature = torch.mean(text_feature, dim=0)
                '''###################################################################
                
                text_feature_img_novel.append(text_feature)
                
            #self.model.to('cpu')
            
        ###############normal
        text_feature_img_novel = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_img_novel], dim=0)
        print('text feature img novel: ', text_feature_img_novel.shape)
        text_feature_img_novel = text_feature_img_novel / text_feature_img_novel.norm(dim=-1, keepdim=True)
        text_feature_avg_novel = torch.sum(text_feature_img_novel, dim=1)
        text_feature_avg_novel = text_feature_avg_novel / text_feature_img_novel.size(1)
        print('text feature avg novel: ', text_feature_avg_novel.shape)
        '''
        #################description
        text_feature_avg_novel = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_img_novel], dim=0)
        print('text feature img novel: ', text_feature_avg_novel.shape)
        '''
        #self.text_feature_avg_novel = text_feature_avg_novel.half()
        #text_features = torch.cat([text_feature_avg_base, text_feature_avg_novel], dim=0)
        #text_features = text_feature_avg_base 
        #text_features = text_feature_avg_novel
        #text_features = torch.cat([text_feature_avg_base, text_feature_avg_novel], dim=0)    
        ################################################################################

        
        
        
        #self.text_features_noavg = text_feature_img_base              #[num_class, num_attribute, width]
        #self.text_features_noavg = text_feature_img_novel             #[num_class, num_attribute, width]
        #self.text_features_noavg = torch.cat([text_feature_img_base, text_feature_img_novel], dim=0)
        
        #self.text_features_noavg = text_feature_img_base + text_feature_img_novel        #list
        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            self.text_features = text_feature_img_base #text_feature_avg_base   #
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        elif self.cfg.DATASET.SUBSAMPLE_CLASSES == 'new':
            self.text_features = text_feature_img_novel #text_feature_avg_novel  #
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        elif self.cfg.DATASET.SUBSAMPLE_CLASSES == 'all':
            self.text_features = torch.cat([text_feature_avg_base, text_feature_avg_novel], dim=0) #torch.cat([text_feature_img_base, text_feature_img_novel], dim=0)  #
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.clip_model = clip_model
        

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        #text_features_noavg = self.text_features_noavg         # [num_class, num_attribute, width]
        '''
        logits_prompt = []
        text_features = self.text_features.permute(1, 0, 2)     #[num_class, num_attribute, width] ---> [num_attribute, num_class, width]
        num_attribute = text_features.size(0)
        image_features = image_features.unsqueeze(0).expand(num_attribute, -1, -1) # [batch_size, width] ---> [num_attribute, batch_size, width]
        logits_prompt = logit_scale * image_features @ text_features.transpose(1, 2)
        logits_prompt = logits_prompt.permute(1, 2, 0)
        logits_scores = logits_prompt.permute(2, 0, 1)
        list_prompt = [t for t in logits_scores]
        return list_prompt
        '''

        logits = logit_scale * image_features @ self.text_features.t()
        return logits
        

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        '''
        print('im in!!!')

        print('save visual feature and score!!!')

        data_loader = self.test_loader

        score_path = '/home/tiggers/mydisk/lql/datasets/oxford_flower/score/ours2/new/test/score.pth'
        score_feature_path = '/home/tiggers/mydisk/lql/datasets/oxford_flower/score/ours2/new/test/feature.pth'
        score_label_path = '/home/tiggers/mydisk/lql/datasets/oxford_flower/score/ours2/new/test/score_label.pth'
        feature_label_path = '/home/tiggers/mydisk/lql/datasets/oxford_flower/score/ours2/new/test/feature_label.pth'

        count = 0

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input = batch['img']
            label_img = batch['label']
            impath = batch['impath']
            self.save_feature(input, label_img, impath)
            self.save_score(input, label_img, impath)
            count = count + 1

            if count == 500:
                print('saving!!!')
                count = 0
                self.score_existing_data = torch.stack(self.score_existing_data, dim=0)
                self.feature_existing_data = torch.stack(self.feature_existing_data, dim=0)
                self.label_score = torch.stack(self.label_score, dim=0)
                self.label_feature = torch.stack(self.label_feature, dim=0)

                if os.path.exists(score_path):
                    score_exist = torch.load(score_path)
                    score_exist = torch.cat([score_exist.cpu(), self.score_existing_data.cpu()], dim=0)
                    
                    torch.save(score_exist.cpu(), score_path)
                else:
                    torch.save(self.score_existing_data.cpu(), score_path)

                if os.path.exists(score_feature_path):
                    feature_exist = torch.load(score_feature_path)
                    feature_exist = torch.cat([feature_exist.cpu(), self.feature_existing_data.cpu()], dim=0)
                    torch.save(feature_exist.cpu(), score_feature_path)
                else:
                    torch.save(self.feature_existing_data.cpu(), score_feature_path)

                if os.path.exists(score_label_path):
                    score_label_exist = torch.load(score_label_path)
                    score_label_exist = torch.cat([score_label_exist.cpu(), self.label_score.cpu()], dim=0)
                    torch.save(score_label_exist.cpu(), score_label_path)
                else:
                    torch.save(self.label_score.cpu(), score_label_path)

                if os.path.exists(feature_label_path):
                    feature_label_exist = torch.load(feature_label_path)
                    feature_label_exist = torch.cat([feature_label_exist.cpu(), self.label_score.cpu()], dim=0)
                    torch.save(feature_label_exist.cpu(), feature_label_path)
                else:
                    torch.save(self.label_feature.cpu(), feature_label_path)


                self.score_existing_data = []
                self.feature_existing_data = []
                self.label_feature = []
                self.label_score = []
                score_exist = []
                feature_exist = []

        self.score_existing_data = torch.stack(self.score_existing_data, dim=0)
        self.feature_existing_data = torch.stack(self.feature_existing_data, dim=0)
        self.label_score = torch.stack(self.label_score, dim=0)
        self.label_feature = torch.stack(self.label_feature, dim=0)

        if os.path.exists(score_path):
            score_exist = torch.load(score_path)
            score_exist = torch.cat([score_exist.cpu(), self.score_existing_data.cpu()], dim=0)
            torch.save(score_exist.cpu(), score_path)
        else:
            torch.save(self.score_existing_data.cpu(), score_path)

        if os.path.exists(score_feature_path):
            feature_exist = torch.load(score_feature_path)
            feature_exist = torch.cat([feature_exist.cpu(), self.feature_existing_data.cpu()], dim=0)
            torch.save(feature_exist.cpu(), score_feature_path)
        else:
            torch.save(self.feature_existing_data.cpu(), score_feature_path)

        if os.path.exists(score_label_path):
            score_label_exist = torch.load(score_label_path)
            score_label_exist = torch.cat([score_label_exist.cpu(), self.label_score.cpu()], dim=0)
            torch.save(score_label_exist.cpu(), score_label_path)
        else:
            torch.save(self.label_score.cpu(), score_label_path)

        if os.path.exists(feature_label_path):
            feature_label_exist = torch.load(feature_label_path)
            feature_label_exist = torch.cat([feature_label_exist.cpu(), self.label_score.cpu()], dim=0)
            torch.save(feature_label_exist.cpu(), feature_label_path)
        else:
            torch.save(self.label_feature.cpu(), feature_label_path)
        '''

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        
        
        
        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label_img = self.parse_batch_test(batch)
            
            #################################################################################
            '''
            logits_prompt = self.model_inference(input)
            #self.model = self.model.to('cuda')
            
            logits_prompt_tensor = torch.stack(logits_prompt)
            logits_prompt_tensor = logits_prompt_tensor.permute(1, 0, 2)      #[num_attribute, batch, num_cls] ---> [batch, num_attribute, num_cls]
            
            self.evaluator.vote(logits_prompt_tensor, label_img)
            
            for att, val in enumerate(logits_prompt):    
               self.evaluator_prompt.process(logits_prompt[att], label_img, att)
            '''
            ##################################################################################avg
            logits = self.model_inference(input)
            self.evaluator.process(logits, label_img)
           

        #results_prompt = self.evaluator_prompt.evaluate()
        results_img = self.evaluator.evaluate()

        for k, v in results_img.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        for k, v in results_img.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results_img.values())[0]
    

    def save_score(self, img, label, path):

        score = self.infernce_score(img)


        single_item = {}
        i = 0

        for index, single_path in enumerate(path):
            self.score_existing_data.append(score[index])
            self.label_score.append(label[index])            
        

    def infernce_score(self, img):
        score = self.score(img)

        return score
    
    def score(self, image):
        logit_scale = self.clip_model.logit_scale.exp()
        logit_scale = logit_scale.exp()
        image_features = self.inference_feature(image)#self.clip_model.encode_image(image.type(self.clip_model.dtype).to('cuda'))
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_prompt = []
        #text_features = torch.cat([self.text_feature_base, self.text_feature_novel], dim=0)
        
        text_features = self.text_features  # [num_class, num_attribute, width]
        text_features = text_features.permute(1, 0, 2) # [num_class, num_attribute, width] ---> [num_attribute, num_class, width]  * [batch_size, width]
        #text_features = self.text_feature_base.permute(1, 0, 2)     #[num_class, num_attribute, width] ---> [num_attribute, num_class, width]
        num_attribute = text_features.size(0)
        #image_features = image_features.unsqueeze(0).expand(num_attribute, -1, -1) # [batch_size, width] ---> [num_attribute, batch_size, width]
        '''
        for i in range(text_features.shape[0]):
            single_prompt_feature = image_features[i, :, :]        #prompt attribute_i [batch, width]

            single_text_feature = text_features[i, :, :]            #text attribute_i   [num_class, width]

            single_prompt_feature = single_prompt_feature / single_prompt_feature.norm(dim=-1, keepdim=True)
            single_text_feature = single_text_feature / single_text_feature.norm(dim=-1, keepdim=True)
            logit_prompt = logit_scale * single_prompt_feature @ single_text_feature.t()       #logit  [batch_size, num_class]

            logits_prompt.append(logit_prompt) # score
        '''
        score_all = torch.einsum("abc,dc->adb", text_features, image_features)
        #score_all = torch.stack(logits_prompt)     # score = [num_attribute, batch_size, num_class]
        score_all = score_all.permute(1, 0, 2)     # score = [batch_size, num_attribute, num_class]

        return score_all

    def save_feature(self, img, label, path):

        feature = self.inference_feature(img)

        for index, single_path in enumerate(path):
            self.feature_existing_data.append(feature[index])
            self.label_feature.append(label[index])
            
        

    def inference_feature(self, img):
        img = img.to('cuda')
        image_features = self.clip_model.encode_image(img.type(self.clip_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        #self.text_features = mean_text_features
        self.clip_model = clip_model

