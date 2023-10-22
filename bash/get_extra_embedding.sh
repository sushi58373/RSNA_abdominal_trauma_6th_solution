MODEL=1 # 0:seresnext50_32x4d, 1:efficientnetv2s

cd ../src

echo "### [stage2] Get Embeddings by Extravasation Feature Extractor ###"

echo "---> get embedding by < seresnext50_32x4d > model"
python stage2-extra-type1-classifier-type2.py --run_type get_emb

cd ../bash