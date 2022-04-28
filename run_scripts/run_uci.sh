PYPATH="/c/Users/jeffe/anaconda3/envs/visualenv/python"
echo "Creating dataset .csv files"

# $PYPATH create_equivalent_dataset.py

# uci on uci (every)
# zigong on zigong
# uci on zigong (best)
# zigong on uci
# uci+zigong on uci+zigong

printf "\n\n\n##### NOW RUNNING NORMALIZED UCI DATA #####\n\n"

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--model svm 

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--model kmeans 

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--data_balance \
--model svm 

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--data_balance \
--model kmeans 

$PYPATH main.py \
--problem death \
--train_dataset normalizeducidf \
--model dnn \
--loss ce

# $PYPATH main.py \
# --problem death \
# --train_dataset normalizeducidf \
# --data_balance \
# --model dnn \
# --loss ce

# $PYPATH main.py \
# --problem death \
# --train_dataset normalizeducidf \
# --model dnn \
# --loss focal
