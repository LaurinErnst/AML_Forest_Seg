#!/bin/bash
cd ../../../
for i in {0..3}
do
    echo "Run: $i"
    mkdir results
    cd results
    mkdir loss_graphs
    mkdir trained_models
    cd ../
    python src/Satnet/SatnetTraining/train_1.py
    python src/Satnet/SatnetTraining/train_2.py
    python src/Satnet/SatnetTraining/train_3.py
    python src/Satnet/SatnetTraining/train_4.py

    #mv -R results/ results$i
    zip -q -r results$i.zip results
    rm -rf results
done




