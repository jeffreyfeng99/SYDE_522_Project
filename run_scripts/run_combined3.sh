PYPATH="/c/Users/jeffe/anaconda3/envs/visualenv/python"
echo "Creating dataset .csv files"

# $PYPATH create_equivalent_dataset.py

printf "\n\n\n##### NOW RUNNING NORMALIZED UCI DATA #####\n\n"

$PYPATH main.py \
--problem death \
--train_dataset normalizeduciandzigongdf \
--model dnn \
--loss focal



