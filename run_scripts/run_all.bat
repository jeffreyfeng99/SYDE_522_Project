conda activate visualenv

@echo off
python create_equivalent_dataset.py %*
pause

PYPATH="/c/Users/jeffe/anaconda3/envs/visualenv/python"
echo "Creating dataset .csv files"
"C:\\Users\\jeffe\\anaconda3\\envs\\visualenv"
"C:\\Users\\jeffe\\anaconda3\\envs\\visualenv\python" create_equivalent_dataset.py

# uci on uci (every)
# zigong on zigong
# uci on zigong (best)
# zigong on uci
# uci+zigong on uci+zigong

echo "Run 1:"
$PYPATH main.py \
--problem death \
--train_dataset normalized_uci_df \
--data_balance \
--model svm 

$PYPATH main.py \
--problem death \
--train_dataset normalized_uci_df \
--data_balance \
--model kmeans 

# python main.py \
# --problem death \
# --train_dataset normalized_uci_df \
# --model dnn \
# --loss ce

# python main.py \
# --problem death \
# --train_dataset normalized_uci_df \
# --data_balance \
# --model dnn \
# --loss ce

# python main.py \
# --problem death \
# --train_dataset \
# --cross \
# --data_balance \
# --model \
# --loss


# python main.py \
# --problem death \
# --train_dataset \
# --cross \
# --data_balance \
# --model \
# --loss

# python main.py \
# --problem death \
# --train_dataset \
# --cross \
# --data_balance \
# --model \
# --loss


# python main.py \
# --problem death \
# --train_dataset \
# --cross \
# --data_balance \
# --model \
# --loss


# python main.py \
# --problem death \
# --train_dataset \
# --cross \
# --data_balance \
# --model \
# --loss