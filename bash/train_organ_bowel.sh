
cd ../src
echo "### [stage2] Train Organ Model : (liver, spleen, left kidney, right kidney) - train kfolds ###"
python stage2-organ-type1.py --loss_type ll --fold 5

echo "### [stage2] Train Bowel Model : (Bowel) ###"
python stage2-bowel-type1.py

cd ../bash