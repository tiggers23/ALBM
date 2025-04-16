import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import matplotlib.pyplot as plt
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import time
import datetime

import numpy as np
from collections import Counter
import os
import math

from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

_tokenizer = _Tokenizer()

NUM_CLS = {
    'Food101_base': 51, # 101
    'Food101_all': 101,
    'OxfordFlowers_base': 51, # 102
    'OxfordFlowers_all': 102,
    'OxfordPets_base': 19, # 37
    'OxfordPets_all': 37,
    'FGVCAircraft_base': 50, # 100
    'FGVCAircraft_all': 100, 
    'DescribableTextures_base': 24, # 47
    'DescribableTextures_all': 47,
    'CUB_base': 100, # 200
    'CUB_all': 200,
    'cifar10_base': 5,
    'cifar10_all': 10,
    'cifar100_base': 50, # 50
    'cifar100_all': 100
}

NUM_ATTR = {
    'Food101': 29,       
    'OxfordFlowers': 26, 
    'OxfordPets': 12,   
    'FGVCAircraft': 23,    
    'DescribableTextures': 33, 
    'CUB': 37,           
    'cifar10': 11,
    'cifar100': 21,
}

DATASET_PATH = {
    'Food101': 'food101',

    'OxfordFlowers': 'oxford_flower',

    'OxfordPets': 'oxfordpets',

    'FGVCAircraft': 'aircraft',

    'DescribableTextures': 'dtd/dtd',

    'CUB': 'cub',

    'cifar10': 'CIFAR10',

    'cifar100': 'CIFAR100',

}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details, NUM_ATTR[cfg.DATASET.NAME])


    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        print("prompts shape:", prompts.shape)
        print("tokenized prompts shape:", tokenized_prompts.shape)
        print("positional_embedding shape", self.positional_embedding.type(self.dtype).shape)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class TextEncoder_Attribute(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transfomer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts):
        text_attribute_feature = []
        #tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        print("positional_embedding shape:", self.positional_embedding.type(self.dtype).shape)
        #for prompt in prompts:
        print("prompt shape:", prompts.shape)
        x = self.token_embedding(prompts).type(self.dtype)
        x = x.unsqueeze(0).expand(13, -1, -1)
        print("expanded prompt shape:", x.shape)
        x = x + self.positional_embedding.type(self.dtype)
        print("x shape:", x.shape)
        x = x.permute(1, 0, 2)
        x = x.float() 
        #x = x.to(torch.device('cuda'))
        outputs = self.transfomer(x)
        x = outputs[0]
        print("after transfomer x:", x.shape)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), prompts[prompts.index(x)].argmax(dim=-1)] @ self.text_projection
        text_attribute_feature.append(x)

        return text_attribute_feature



class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)                   
        n_ctx = cfg.TRAINER.MAPLE.N_CTX           
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT     
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"    #保证prompt_depth >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        #初始化text prompt
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")


        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts    复合提示
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        print("ctx shape:", ctx.shape)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
            #print("vision_prompt:", layer(self.compound_prompts_text[index]).size())
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class VisionpromptLeaner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = NUM_ATTR[cfg.DATASET.NAME]  #len(attributename_base)       #应为attribute的数量
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        attributenames = NUM_ATTR[cfg.DATASET.NAME]
        n_ctx = 16

        prompt_init = clip_model.visual.class_embedding
        prompt_init_position_embedding = clip_model.visual.positional_embedding[0]
        self.prompt_init = prompt_init.detach() + prompt_init_position_embedding.detach()
        



        #initialize vision prompt
        if not cfg.TRAINER.MAPLE.INITED:
        #    return 0
            print("Initialize prompt_prefix use clip_model classs embedding!")
            ctx_vectors = self.prompt_init.unsqueeze(0).expand((NUM_ATTR[cfg.DATASET.NAME]), -1)
            #print("shape of inited prompts")
        else:
            print("Random Initialize prompt_prefix!")
            ctx_vectors = torch.empty(NUM_ATTR[cfg.DATASET.NAME], 768, dtype=dtype)  # n_ctx替换为设置的长度   ctx_dim替换为768
            nn.init.normal_(ctx_vectors, std=0.02)
        
        

        self.vision_prompts = nn.Parameter(ctx_vectors.to('cuda'))          



        self.init_weight = torch.zeros(NUM_CLS["{}_base".format(cfg.DATASET.NAME)], NUM_ATTR[cfg.DATASET.NAME])
        nn.init.kaiming_normal_(self.init_weight)
        self.mat_0 = nn.Parameter(self.init_weight.to('cuda').to(dtype).clone())
        self.mat =self.mat_0
        
        prompt_prefix= " ".join(["X"] * NUM_ATTR[cfg.DATASET.NAME])

        print("Attribute Vision Prompt")
        print(f"Number of attribute vision prompt:{NUM_ATTR[cfg.DATASET.NAME]}")   #n_ctx替换为num_attribute

        self.ctx = ctx_vectors

        self.n_cls = n_cls
        self.n_ctx = n_ctx


    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        return ctx, self.vision_prompts
    

class CustomCLIP_Attribute(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VisionpromptLeaner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_Attribute(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model

        self.cfg = cfg
        print(cfg.DATASET.NAME)
        if cfg.DATASET.NAME == 'cifar10':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10-base.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10_ours_6_label_special_distribution.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR10/cifar10_ours_6_classname_special_distribution.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'cifar100':
            self.base_file = '/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar100-base.txt'
            self.novel_file = '/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar100-novel.txt'
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/CIFAR100/cifar-100-python/cifar_ours_classname.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'CUB':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/cub/cub-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/cub/cub-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_labo_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/cub/cub_labo_classname.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'DescribableTextures':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/dtd/dtd/dtd-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/dtd/dtd/DTD_ours_classname.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'FGVCAircraft':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/aircraft/fgvc-aircraft-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/aircraft/aircraft_labo_classname.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'Food101':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/food101/food-101/food101-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/food101/food-101/food_all_visual_noattributename_classname.pt', map_location='cpu', encoding='latin1')
        
        elif cfg.DATASET.NAME == 'OxfordFlowers':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/oxford_flower/flowers102-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxford_flower/flower_labo_classname.pt', map_location='cpu', encoding='latin1')

        elif cfg.DATASET.NAME == 'OxfordPets':
            self.base_file = "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-base.txt"
            self.novel_file = "/home/tiggers/mydisk/lql/datasets/oxfordpets/pets-novel.txt"
            self.gpt_text_after = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_label.pt', map_location='cpu', encoding='latin1')
            self.gpt_text_pre = torch.load('/home/tiggers/mydisk/lql/datasets/oxfordpets/pets_ours_classname.pt', map_location='cpu', encoding='latin1')

        if cfg.DATASET.SUBSAMPLE_CLASSES == "base":
            self.base_or_novel = "base"
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "new":
            self.base_or_novel = "novel"
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "all":
            self.base_or_novel = "all"

        text_feature_all_base = []
        text_feature_all_novel = []
        
        if self.base_or_novel == 'base':
            with open(self.base_file, 'r') as file:
                base_class_name = [line.strip() for line in file.readlines()]

            #comput the text feature to align prompt feature with text feature
            for i in range(len(base_class_name)):
                gpt_sentence = self.gpt_text_after[i]
            
                gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])

                gpt_sentence = gpt_sentence.to('cuda')
                self.model = self.model.to('cuda')

                with torch.no_grad():
                    text_feature = self.model.encode_text(gpt_sentence)
                    text_feature_all_base.append(text_feature)
                
                self.model.to('cpu')

            text_feature_base = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_all_base], dim=0)
            self.text_feature_base = text_feature_base



        elif self.base_or_novel == 'novel':
            with open(self.novel_file, 'r') as file:
                novel_class_name = [line.strip() for line in file.readlines()]
            for i in range(len(novel_class_name)):
                gpt_sentence = self.gpt_text_after[i + ( NUM_CLS["{}_base".format(cfg.DATASET.NAME)] )]
            
                gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])

                gpt_sentence = gpt_sentence.to('cuda')
                self.model = self.model.to('cuda')

                with torch.no_grad():
                    text_feature = self.model.encode_text(gpt_sentence)
                    text_feature_all_novel.append(text_feature)
                
                self.model.to('cpu')

            text_feature_novel = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_all_novel], dim=0)
            self.text_feature_novel = text_feature_novel


        elif self.base_or_novel == 'all':
            with open(self.base_file, 'r') as file:
                base_class_name = [line.strip() for line in file.readlines()]
            #comput the text feature to align prompt feature with text feature
            for i in range(len(base_class_name)):
                gpt_sentence = self.gpt_text_after[i]
            
                gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])

                gpt_sentence = gpt_sentence.to('cuda')
                self.model = self.model.to('cuda')

                with torch.no_grad():
                    text_feature = self.model.encode_text(gpt_sentence)
                    text_feature_all_base.append(text_feature)
                
                self.model.to('cpu')

            text_feature_base = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_all_base], dim=0)
            self.text_feature_base = text_feature_base


            text_feature_img_novel = []
            with open(self.novel_file, 'r') as file:
                novel_class_name = [line.strip() for line in file.readlines()]

            for i in range(len(novel_class_name)):
                gpt_sentence = self.gpt_text_after[i + ( NUM_CLS["{}_base".format(cfg.DATASET.NAME)] )]
            
                gpt_sentence = torch.cat([clip.tokenize(s) for s in gpt_sentence])

                gpt_sentence = gpt_sentence.to('cuda')
                self.model = self.model.to('cuda')

                with torch.no_grad():
                    text_feature = self.model.encode_text(gpt_sentence)
                    text_feature_all_novel.append(text_feature)
                
                self.model.to('cpu')

            text_feature_novel = torch.cat([text_feature.unsqueeze(0) for text_feature in text_feature_all_novel], dim=0)
            self.text_feature_novel = text_feature_novel

            self.text_feature_base = self.text_feature_base / self.text_feature_base.norm(dim=-1, keepdim=True)
            self.text_feature_novel = self.text_feature_novel / self.text_feature_novel.norm(dim=-1, keepdim=True)
            self.text_feature_all = torch.cat([self.text_feature_base, self.text_feature_novel], dim=0)

        if self.base_or_novel == 'base':
            self.text_feature_all = self.text_feature_base
            self.text_feature_all = self.text_feature_all / self.text_feature_all.norm(dim=-1, keepdim=True)
        elif self.base_or_novel == 'novel':
            self.text_feature_all = self.text_feature_novel
            self.text_feature_all = self.text_feature_all / self.text_feature_all.norm(dim=-1, keepdim=True)
        else:
            self.text_feature_all = self.text_feature_all
            self.text_feature_all = self.text_feature_all / self.text_feature_all.norm(dim=-1, keepdim=True)

    def forward(self, label, img):
        logit_scale = self.logit_scale.exp()
        prompts, vision_prompts = self.prompt_learner()
        self.text_feature_all_trans = self.text_feature_all.permute(1, 0, 2)
        
        if self.prompt_learner.training:
            # train prompt(vote)
            if self.prompt_learner.vision_prompts.requires_grad is True:
                prompt_features, image_features = self.image_encoder(img.type(self.dtype), vision_prompts, NUM_ATTR[self.cfg.DATASET.NAME])

                prompt_features = prompt_features.permute(1, 0, 2)             # [batch_size, num_attribute, width] -> [num_attribtue, batch_size, width]

                logits_prompt = logit_scale * prompt_features @ self.text_feature_all_trans.transpose(1, 2) # scores = [num_attribute, batch_size, num_cls]
                logits_prompt = logits_prompt.permute(1, 2, 0) # [num_attribute, batch_size, num_cls] ---> [batch_size, num_cls, num_attribute]

                logits_prompt = logits_prompt.permute(2, 0, 1)
                loss_prompt = torch.tensor(0)
                for index, item in enumerate(logits_prompt):
                    loss_prompt = loss_prompt + F.cross_entropy(item, label)

                loss_prompt = loss_prompt / NUM_ATTR[self.cfg.DATASET.NAME]
                return loss_prompt
            
            # train matrix
            elif self.prompt_learner.mat.requires_grad is True:
                prompt_features, image_features = self.image_encoder(img.type(self.dtype), vision_prompts, NUM_ATTR[self.cfg.DATASET.NAME])

                prompt_features = prompt_features.permute(1, 0, 2)             # [batch_size, num_attribute, width] -> [num_attribtue, batch_size, width]

                logits_prompt = logit_scale * prompt_features @ self.text_feature_all_trans.transpose(1, 2) # scores = [num_attribute, batch_size, num_cls]
                logits_prompt = logits_prompt.permute(1, 2, 0) # [num_attribute, batch_size, num_cls] ---> [batch_size, num_cls, num_attribute]

                score_class = logits_prompt * self.prompt_learner.mat.unsqueeze(0)
                score_class = score_class.sum(dim=2)

                loss = F.cross_entropy(score_class, label.long())
                return loss


        else: # test
            if self.prompt_learner.vision_prompts.requires_grad is True:
                prompt_features, image_features = self.image_encoder(img.type(self.dtype), vision_prompts, NUM_ATTR[self.cfg.DATASET.NAME])
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

                prompt_features = prompt_features.permute(1, 0, 2)             # [batch_size, num_attribute, width] -> [num_attribtue, batch_size, width]

                logits_prompt = logit_scale * prompt_features @ self.text_feature_all_trans.transpose(1, 2) # scores = [num_attribute, batch_size, num_cls]
                logits_prompt = logits_prompt.permute(1, 2, 0)

                logits_scores = logits_prompt.permute(2, 0, 1)
                list_scores = [t for t in logits_scores]
                return list_scores, logits_prompt
            elif self.prompt_learner.mat.requires_grad is True:
                prompt_features, image_features = self.image_encoder(img.type(self.dtype), vision_prompts, NUM_ATTR[self.cfg.DATASET.NAME])
                prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

                prompt_features = prompt_features.permute(1, 0, 2)             # [batch_size, num_attribute, width] -> [num_attribtue, batch_size, width]

                logits_prompt = logit_scale * prompt_features @ self.text_feature_all_trans.transpose(1, 2) # scores = [num_attribute, batch_size, num_cls]
                logits_prompt = logits_prompt.permute(1, 2, 0) # [batch_size, num_cls, num_attribute]

                scores = logits_prompt
                if self.base_or_novel == "base":

                    score_class = scores * self.prompt_learner.mat.unsqueeze(0)
                    score_class = score_class.sum(dim=2)

                    scores = scores.permute(0, 2, 1)
                    scores = scores.permute(1, 0, 2)
                    list_scores = [t for t in scores]

                    return list_scores, score_class
            
                elif self.base_or_novel == "novel":
                    self.model = self.model.to('cuda')
                    with open(self.base_file, 'r') as file:
                        base_class_name = [line.strip() for line in file.readlines()]
                    base_class_sentence = ['a photo of {}'.format(name) for name in base_class_name]
                    base_class_sentence = torch.cat([clip.tokenize(s) for s in base_class_sentence])
                    base_class_sentence = base_class_sentence.to('cuda')
                    with torch.no_grad():
                        base_name_feature = self.model.encode_text(base_class_sentence)
                
                    with open(self.novel_file, 'r') as file:
                        novel_class_name = [line.strip() for line in file.readlines()]
                    novel_class_sentence = ['a photo of {}'.format(name) for name in novel_class_name]
                    novel_class_sentence = torch.cat([clip.tokenize(s) for s in novel_class_sentence])
                    novel_class_sentence = novel_class_sentence.to('cuda')
                    with torch.no_grad():
                        novel_name_feature = self.model.encode_text(novel_class_sentence)

                    self.novel_mat = torch.zeros(50, NUM_ATTR[self.cfg.DATASET.NAME])
                    for i, single_novel_name_feature in enumerate(novel_name_feature):
                        similarity = F.cosine_similarity(base_name_feature, single_novel_name_feature)
                        weights = F.softmax(similarity, dim=-1)
                        weights = weights.unsqueeze(1)
                        #print(weights.shape)
                        x = weights * self.prompt_learner.mat
                        self.novel_mat[i] = x.sum(dim=0)
                    self.novel_mat = self.novel_mat.to('cuda')
                
                    scores = scores.permute(0, 2, 1)
                    score_class = scores * self.novel_mat.unsqueeze(0)
                    score_class = score_class.sum(dim=2)

                    scores = scores.permute(0, 2, 1)
                    scores = scores.permute(1, 0, 2)
                    list_scores = [t for t in scores]

                    return list_scores, score_class
            
                elif self.base_or_novel == 'all':
                    score_class = scores * self.prompt_learner.mat.unsqueeze(0)
                    score_class = score_class.sum(dim=2)

                    scores = scores.permute(0, 2, 1)
                    scores = scores.permute(1, 0, 2)
                    list_scores = [t for t in scores]
                
                    return list_scores, score_class

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)


        self.evaluator_prompt = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.evaluator_img = build_evaluator(cfg, lab2cname=self.lab2cname)
        #self.loss = []
        self.score_existing_data = []
        self.feature_existing_data = []
        self.label_feature = []
        self.label_score = []
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        #print("Building custom CLIP")
        #self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Building custom CLIP_Attribute!")
        self.model = CustomCLIP_Attribute(cfg, classnames, clip_model)

        self.clip_model = clip_model

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                print(f'parameter name: {name}, parameter device: {param.device}')
        print(f"Parameters to be updated: {enabled}")
        

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #self.model = nn.DataParallel(self.model)

    def reset_stage2(self):
        cfg = self.cfg
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self._scheds["MultiModalPromptLearner"] = self.sched

    def build_optimizer_stage2(model, optim_cfg, param_groups=None):
        """A function wrapper for building an optimizer.

            Args:
                model (nn.Module or iterable): model.
                optim_cfg (CfgNode): optimization config.
                param_groups: If provided, directly optimize param_groups and abandon model
        """
        lr = 0.0006
        weight_decay = optim_cfg.WEIGHT_DECAY
        momentum = optim_cfg.MOMENTUM
        sgd_dampening = optim_cfg.SGD_DAMPNING
        sgd_nesterov = optim_cfg.SGD_NESTEROV
        staged_lr = optim_cfg.STAGED_LR
        new_layers = optim_cfg.NEW_LAYERS
        base_lr_mult = optim_cfg.BASE_LR_MULT

        if param_groups is None:
            if staged_lr:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "When staged_lr is True, model given to "
                        "build_optimizer() must be an instance of nn.Module"
                    )

                if isinstance(model, nn.DataParallel):
                    model = model.module

                if isinstance(new_layers, str):
                    if new_layers is None:
                        warnings.warn("new_layers is empty (staged_lr is useless)")
                    new_layers = [new_layers]

                base_params = []
                base_layers = []
                new_params = []

                for name, module in model.named_children():
                    if name in new_layers:
                        new_params += [p for p in module.parameters()]
                    else:
                        base_params += [p for p in module.parameters()]
                        base_layers.append(name)

                param_groups = [
                    {
                        "params": base_params,
                        "lr": lr * base_lr_mult
                    },
                    {
                        "params": new_params
                    },
                ]

            else:
                if isinstance(model, nn.Module):
                    param_groups = model.parameters()
                else:
                    param_groups = model

        optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=sgd_dampening,
                nesterov=sgd_nesterov,
        )

        return optimizer

    def forward_backward(self, batch, stage=None):
        cfg = self.cfg
        image, label = self.parse_batch_train(batch)
        model = self.model
        model = model.to('cuda')

        if stage == 'stage1':
            optim = self.optim #optimizer
            scaler = self.scaler
        elif stage == 'stage2':
            optim = self.build_optimizer_stage2
            scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(label, image)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:

            loss = model(label, image)
            loss = loss.to('cuda')
            model = model.to('cuda')

            optim.zero_grad()
            loss.backward()
            optim.step()


        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]


        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]


        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    

    def before_train(self):
        cfg = self.cfg
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

        self.set_model_mode("eval")
        
        data_loader = self.test_loader

        print(f"Evaluate on the *test* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):

            image, label = self.parse_batch_test(batch)
            logits_prompt, logits_img = self.model_inference(label, image)
            #self.model = self.model.to('cuda')
            logits_prompt_tensor = torch.stack(logits_prompt)
            logits_prompt_tensor = logits_prompt_tensor.permute(1, 0, 2)      #[num_attribute, batch, num_cls] ---> [batch, num_attribute, num_cls]
            
            self.evaluator_img.vote(logits_prompt_tensor, label)
            for att, val in enumerate(logits_prompt):   
                self.evaluator_prompt.process(logits_prompt[att], label, att)

        results_prompt = self.evaluator_prompt.evaluate()
        results_img = self.evaluator_img.evaluate()
        
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        self.evaluator_img.reset()
        self.evaluator_prompt.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        cfg = self.cfg
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if self.model.prompt_learner.vision_prompts.requires_grad is True:
                image, label = self.parse_batch_test(batch)
                logits_prompt, logits_img = self.model_inference(label, image)
                self.model = self.model.to('cuda')
                logits_prompt_tensor = torch.stack(logits_prompt)
                logits_prompt_tensor = logits_prompt_tensor.permute(1, 0, 2)      #[num_attribute, batch, num_cls] ---> [batch, num_attribute, num_cls]
                self.evaluator_img.vote(logits_prompt_tensor, label)
                for att, val in enumerate(logits_prompt):    
                    self.evaluator_prompt.process(logits_prompt[att], label, att)
            elif self.model.prompt_learner.mat.requires_grad is True:
                image, label = self.parse_batch_test(batch)
                logits_prompt, logits_img = self.model_inference(label, image)
                self.model = self.model.to('cuda')
                logits_prompt_tensor = torch.stack(logits_prompt)
                logits_prompt_tensor = logits_prompt_tensor.permute(1, 0, 2)      #[num_attribute, batch, num_cls] ---> [batch, num_attribute, num_cls]
                self.evaluator_img.process(logits_img, label)

        #results_prompt = self.evaluator_prompt.evaluate()
        results_img = self.evaluator_img.evaluate()

        for k, v in results_img.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        for k, v in results_img.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results_img.values())[0]
    

    def model_inference(self, label, image):

        return self.model(label, image)

    def model_inference_w(self, label, features, scores):

        return self.model(label, features, scores)
    


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        do_test = False
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if do_test and self.cfg.TEST.FINAL_MODEL == "last_step":
            curr_result = self.test(split="val")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def train(self):
        """Generic training loops."""
        self.only_prompt_epoch = 5
        stage = 'stage1'
        flag = 0

        self.before_train()
        self.model.prompt_learner.vision_prompts.requires_grad = False
        self.model.prompt_learner.mat_0.requires_grad = True
        for self.epoch in range(self.start_epoch, self.max_epoch):
            if (self.epoch >= self.only_prompt_epoch-1) and (flag == 0):
                self.model.prompt_learner.mat_0.requires_grad = True
                self.model.prompt_learner.vision_prompts.requires_grad = False
                stage = 'stage2'
                self.reset_stage2()
                flag = 1

            self.before_epoch()
            self.run_epoch(stage)
            self.after_epoch()

        self.after_train()
    

    def run_epoch(self, stage):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, stage)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()