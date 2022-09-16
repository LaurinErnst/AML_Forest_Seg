#!/bin/bash
cd ../../../
for i in {0..5}
do
    echo "Run: $i"
    mkdir results
    cd results
    mkdir loss_graphs
    mkdir trained_models
    cd ../
    python src/Unet/UNet_training/train_1.py
    python src/Unet/UNet_training/train_2.py
    python src/Unet/UNet_training/train_3.py
    python src/Unet/UNet_training/train_4.py

    #mv -R results/ results$i
    zip -q -r results$i.zip results
    rm -rf results
done




