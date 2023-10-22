MODEL=0 # 0:seresnext50_32x4d, 1:efficientnetv2s

cd ../src

echo "### [stage2] Train Extravasation Feature Extractor ###"

if [ $MODEL -eq 0 ]
then
    echo "---> train < seresnext50_32x4d > model"
    python stage2-extra-type1-classifier-type2-mask.py --run_type train
elif [ $MODEL -eq 1 ]
then
    echo "---> train < efficientnetv2s > model"
    python stage2-extra-type1-classifier-type2-mask-effnet.py --run_type train
else
    echo "check MODEL number to train correctly"
fi

cd ../bash