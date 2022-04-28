PYPATH="/c/Users/jeffe/anaconda3/envs/visualenv/python"
echo "Creating dataset .csv files"

# $PYPATH create_equivalent_dataset.py

printf "\n\n\n##### NOW RUNNING NORMALIZED UCI DATA #####\n\n"

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--model dnn \
--cross \
--validate \
--saved_dir ./output/04272022_fullrunv1_focalfix

printf "\n\n\n##### NOW RUNNING NORMALIZED ZIGONG DATA #####\n\n"

$PYPATH main.py \
--problem death \
--train_dataset normalizedzigongdf \
--model dnn \
--cross \
--validate \
--saved_dir ./output/04272022_fullrunv1_focalfix

