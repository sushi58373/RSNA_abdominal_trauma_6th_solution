
cd ../src/make_dataframe

echo '### [1/6] df_seg.csv dataframe ###'
python df_seg.py

echo '### [2/6] train_df.csv dataframe ###'
python train_df.py

echo '### [3/6] generate png_files dictionary ###'
python generate_png_files_dict.py

echo '### [4/6] dcm_number.csv dataframe ###'
python dcm_number.py

echo '### [5/6] inverse.csv dataframe ###'
source inverse.py

echo '### [6/6] extra_bbox.csv dataframe : stride 5 df, stride pos df ###'
python extra_bbox.py

cd ../../bash