# multi-domain_NER

多领域NER。

## 文件目录

| 文件 | 描述 |
| :----: | :----: |
| _bert-base-chinese_ | BERT预训练模型文件（pytorch） |
| _data_ | 数据（train, dev, test, tags） |
| _ner.py_ | 多领域命名实体识别（pytorch） |
 
## 可选参数
| 参数 | 描述 | 解释 |
| :---- | :---- |
|-h, --help | show this help message and exit | |
|--train_file | TRAIN_FILE The training file path. | 训练数据 |
|--dev_file DEV_FILE |  The development file path. | 验证数据 |
|--test_file TEST_FILE | The testing file path. | 测试数据 |
|--class_file CLASS_FILE | The testing file path. | 标签数据 |
|--output_dir OUTPUT_DIR | The output folder path. | 输出文件夹 |
|--train_batch_size TRAIN_BATCH_SIZE | The number of sentences contained in a batch during training. | 训练的一批句子数 |
|--test_batch_size TEST_BATCH_SIZE |The number of sentences contained in a batch during testing. |验证或测试的一批句子数 |
|--epochs EPOCHS  | Total number of training epochs to perform. | 训练最大轮数 |
|--learning_rate LEARNING_RATE | The initial learning rate for Adam. | 学习率 |
|--max_len MAX_LEN | The Maximum length of a sentence. | 句子最大长度（如果实际句子过长则按照split集切分） |
|--keep_last_n_checkpoints KEEP_LAST_N_CHECKPOINTS | Keep the last n checkpoints. | 保留最后的几轮模型 | 
|--warmup_proportion WARMUP_PROPORTION |Proportion of training to perform linear learning rate warmup for. | warmup |
|--split SPLIT | Characters that segments a sentence. | 句子可以切分的字符（如标点） |
|--tensorboard_dir TENSORBOARD_DIR | The data address of the tensorboard. | Tensorboard路径 |
|--bert_config_file BERT_CONFIG_FILE | The config json file corresponding to the pre-trained BERT model. This specifies the model architecture. | BERT预训练模型 |
|--cpu  | Whether to use CPU, if not and CUDA is avaliable can use CPU. | 如果使用CPU |
|--seed SEED | random seed for initialization. | 随机种子 |
