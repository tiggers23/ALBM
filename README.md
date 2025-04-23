# ALBM

Code for the CVPR2025 paper ["Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability"](https://arxiv.org/pdf/2503.20301)

# Set up environments

We run our experiments using Python 3.9.19 . You can install the required packages using:

```
conda create --name ALBM python=3.9.19
conda activate ALBM
pip install -r requirements.txt
```

# Directories

- `clip/`  saves the code of CLIP model 
- `configs/` saves the config files for all experiments. You can modify the config files to change the system arguments
- `dataset/` saves the code for building the dataset
- `docs/` saves instructions on how to train and test
- `scripts/` saves the script files for training and testing
- `trainer/` saves the models

The results will be saved in `output/` 

# Training and Testing

The relevant script files are saved in [Run,md](docs/RUN.md)



Please cite our paper if you find it useful!

```
@misc{zhang2025attributeformedclassspecificconceptspace,
      title={Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability}, 
      author={Jianyang Zhang and Qianli Luo and Guowu Yang and Wenjing Yang and Weide Liu and Guosheng Lin and Fengmao Lv},
      year={2025},
      eprint={2503.20301},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20301}, 
}
```



