MODEL=1 # 0:seresnext50_32x4d, 1:efficientnetv2s

cd ../src

echo "### [stage2] Get Embeddings by Extravasation Feature Extractor with segmentation head ###"

if [ $MODEL -eq 0 ]
then
    echo "---> get embedding by < seresnext50_32x4d > model"
    python stage2-extra-type1-classifier-type2-mask.py --run_type get_emb
elif [ $MODEL -eq 1 ]
then
    echo "---> get embedding by < efficientnetv2s > model"
    python stage2-extra-type1-classifier-type2-mask-effnet.py --run_type get_emb
else
    echo "check MODEL number to train correctly"
fi

cd ../bash