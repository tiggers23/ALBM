#!/bin/bash

#cd ../..

# custom config
#DATA="/path/to/dataset/folder"
DATA="/home/tiggers/mydisk/lql/datasets"
TRAINER=MaPLe

SEED=$1

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=1000
SUB=new

DATASET_LIST=(cifar10 cifar100 cub dtd fgvc_aircraft food101 oxford_pets)

for DATASET in "${DATASET_LIST[@]}"
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Evaluating model"
        echo "Results are available in ${DIR}. Resuming..."

        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}

    else
        echo "Evaluating model"
        echo "Runing the first phase job and save the output to ${DIR}"

        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done