# Multi-domain Named Entity Recognition (Single Model)

多领域命名实体识别（Pytorch），支持BERT-SPAN-MD模型。该模型将所有领域的数据集混在一起训练，测试时按领域分开。

MD:multi-domain即每个领域有单独的SPAN并通过领域分类器进行集成形成最终的结果。

单领域命名实体识别，请转至[V3.0](https://github.com/newbieyd/multi-domain_NER/releases/tag/v3.0)

## 文件目录

|-bert-base-chinese —— BERT预训练模型文件（pytorch） 

|&emsp;&emsp;|--config.json —— BERT配置文件
          
|&emsp;&emsp;|--pytorch_model.bin —— BERT模型
 
|&emsp;&emsp;|--vocab.txt —— BERT词表
 
|-data —— 数据

|&emsp;&emsp;|--news —— 领域数据
          
|&emsp;&emsp;&emsp;&emsp;|---train.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---dev.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---test.txt

|&emsp;&emsp;&emsp;&emsp;|---class.txt —— Span方法使用的标签格式（实体类别）
                    
|&emsp;&emsp;|--news —— 领域数据

|&emsp;&emsp;&emsp;&emsp;|---train.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---dev.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---test.txt

|&emsp;&emsp;&emsp;&emsp;|---class.txt —— Span方法使用的标签格式（实体类别）

|&emsp;&emsp;|--news —— 领域数据
          
|&emsp;&emsp;&emsp;&emsp;|---train.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---dev.txt
                    
|&emsp;&emsp;&emsp;&emsp;|---test.txt

|&emsp;&emsp;&emsp;&emsp;|---class.txt —— Span方法使用的标签格式（实体类别）

|-multi_domain —— 多领域NER

|&emsp;&emsp;|--data_processor.py —— 数据集构建方法 

|&emsp;&emsp;|--model.py —— 模型方法（SPAN、CRF、SoftMax） 

|&emsp;&emsp;|--multi_domain_ner.py —— 主方法（包括训练、验证、测试等） 

|&emsp;&emsp;|--utils.py —— 一些基础函数 

## 数据格式说明

数据文件（*train*，*dev*，*test*）采用BIO的标注格式，其中每行为一个字符和一个标签（中间以\t分开），空行表示一句话结束。详细可以看*data*中的样例。

SPAN方法使用为实体类别（如*class.txt*），如PER（数据文件中被标记为*B-PER*，*I-PER*，非实体为*O*）。

CRF和SoftMax方法使用数据中全部的标签类别（如*tag.txt*）。

## 环境参数

python        --3.6

torch         --1.4.0 

torchvision   --0.5.0

tensorboard   --2.3.0 

tensorboardX  --2.1

transformers  --3.1.0

tqdm          --4.49.0

注：最新版本安装transformers时，将sentencepiece降到0.1.91版本，否则可能报错。

## 可选参数

| 参数 | 描述 | 解释 |
| :---- | :---- | :---- |
|-h, --help | show this help message and exit | |
|--data_dir DATA_DIR | The data folder path.  | 数据集的根目录 |
|--domain DOMAIN | The domain names (multiple domains separated by \*)  | 领域名也是文件夹名（以\*分开） |
|--train | Training | 训练 |
|--dev |  Development. | 验证 |
|--test | Testing. | 测试 |
|--output_dir OUTPUT_DIR | The output folder path. | 输出文件夹 |
|--model MODEL | The model path. | 验证和测试的模型路径 |
|--architecture {span} | The model architecture of neural network and what decoding method is adopted. | 模型可选{span，crf} |
|--train_batch_size TRAIN_BATCH_SIZE | The number of sentences contained in a batch during training. | 训练的一批句子数 |
|--test_batch_size TEST_BATCH_SIZE |The number of sentences contained in a batch during testing. |验证或测试的一批句子数 |
|--epochs EPOCHS  | Total number of training epochs to perform. | 训练最大轮数 |
|--learning_rate LEARNING_RATE | The initial learning rate for Adam. | 学习率 |
|--crf_lr CRF_LR | The initial learning rate of CRF layer. | CRF层的学习率 |
|--dropout DROPOUT | What percentage of neurons are discarded in the fully connected layers (0 ~ 1). | 全连接层Dropout丢失率 |
|--max_len MAX_LEN | The Maximum length of a sentence. | 句子最大长度（如果实际句子过长则按照split集切分） |
|--keep_last_n_checkpoints KEEP_LAST_N_CHECKPOINTS | Keep the last n checkpoints. | 保留最后的几轮模型 | 
|--warmup_proportion WARMUP_PROPORTION |Proportion of training to perform linear learning rate warmup for. | warmup |
|--split SPLIT | Characters that segments a sentence. | 句子可以切分的字符（如标点） |
|--tensorboard_dir TENSORBOARD_DIR | The data address of the tensorboard. | Tensorboard路径 |
|--domain_loss_rate DOMAIN_LOSS_RATE | Weight of domain loss. | 领域分类器损失比重 |
|--domain_ner_loss_rate DOMAIN_NER_LOSS_RATE | Weight of domain ner loss. | 集成SPAN损失比重 |
|--bert_config_file BERT_CONFIG_FILE | The config json file corresponding to the pre-trained BERT model. This specifies the model architecture. | BERT预训练模型 |
|--cpu  | Whether to use CPU, if not and CUDA is avaliable can use CPU. | 如果使用CPU |
|--seed SEED | random seed for initialization. | 随机种子 |

## 可选参数特殊说明

### 多领域数据格式
--data_dir为数据的跟目录，其下的文件夹的名字同时也是领域名，以\*隔开组成--domain参数。并且每个文件夹下的文件名应为train.txt、dev.txt、test.txt和标签文件class.txt。

### 训练方式
--train --dev --test 分别代表运行方式
>+ 只使用 __--train__ 则只训练到固定轮数，保存为最后的模型 *checkpoint-last.kpl*
>+ 若使用 __--train__ 和 __--dev__ 则会额外域保存在开发集上的最高分数的模型 *checkpoint-best.kpl*
>+ __--test__ 则为测试方式如存在 *checkpoint-best.kpl* 则使用该模型，否则使用 *checkpoint-last.kpl*

### -model作用

如指定则在验证测试的时候使用该模型，否则使用验证集上的最高分时模型（*checkpoint-best.pkl*）,若无验证集则使用模型训练最后的模型（*checkpoint-last.pkl*）

## 脚本样例

./scripts/span_train.sh
