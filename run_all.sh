PYPATH="/c/Users/jeffe/anaconda3/envs/visualenv/python"
echo "Creating dataset .csv files"

# "C:\\Users\\jeffe\\anaconda3\\envs\\visualenv\python" create_equivalent_dataset.py

$PYPATH create_equivalent_dataset.py

# uci on uci (every)
# zigong on zigong
# uci on zigong (best)
# zigong on uci
# uci+zigong on uci+zigong

echo "\n\n##### NOW RUNNING NORMALIZED UCI DATA #####\n"

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

$PYPATH main.py \
--problem death \
--train_dataset normalized_uci_df \
--model dnn \
--loss ce

$PYPATH main.py \
--problem death \
--train_dataset normalized_uci_df \
--data_balance \
--model dnn \
--loss ce

$PYPATH main.py \
--problem death \
--train_dataset normalized_uci_df \
--model dnn \
--loss focal

echo "\n\n##### NOW RUNNING NORMALIZED ZIGONG DATA #####\n"

$PYPATH main.py \
--problem death \
--train_dataset normalized_zigong_df \
--data_balance \
--model svm 

$PYPATH main.py \
--problem death \
--train_dataset normalized_zigong_df \
--data_balance \
--model kmeans 

$PYPATH main.py \
--problem death \
--train_dataset normalized_zigong_df \
--model dnn \
--loss ce

$PYPATH main.py \
--problem death \
--train_dataset normalized_zigong_df \
--data_balance \
--model dnn \
--loss ce

$PYPATH main.py \
--problem death \
--train_dataset normalized_zigong_df \
--model dnn \
--loss focal
