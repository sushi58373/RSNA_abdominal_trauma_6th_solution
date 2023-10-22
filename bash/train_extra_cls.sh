MODEL=0 # 0:seresnext50_32x4d, 1:efficientnetv2s

cd ../src

echo "### [stage2] Train Extravasation Feature Extractor with segmentation head ###"

echo "---> train < seresnext50_32x4d > model"
python stage2-extra-type1-classifier-type2.py --run_type train

cd ../bash