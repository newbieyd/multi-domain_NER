cat ./train.sh

python -u ./ner.py \
--bert_config_file ./bert-base-chinese \
--train_file ./data/train_bio.txt \
--dev_file ./data/dev_bio.txt \
--test_file ./data/test_bio.txt \
--output_dir ./output \
--max_len 200 \
--class_file ./data/class.txt \
--train_batch_size 25 \
--learning_rate 5e-5 \
--epoch 30 \
--test_batch_size 1 \
--tensorboard ./output/logs \
--seed 1

cp ./train.sh ./output
