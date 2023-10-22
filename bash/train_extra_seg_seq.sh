

cd ../src
echo "### [stage2] Train Extravasation Sequence Model from feature extractor with segmentation head ###"
python stage2-extra-type1-sequence-type2-mask.py --eval_type epoch

cd ../bash