
cd ../src
echo "### [stage1] 3D segmentation : train 3D segmentation model with loading cache ###"
python segmentation.py --run_type train
cd ../bash
