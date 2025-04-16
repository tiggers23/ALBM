# Training and Evaluation


### Training time and compute
## ALBM

#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train MaPLe on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 1
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 1

# seed=2
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 2
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 2

# seed=3
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 3
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 3
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```
