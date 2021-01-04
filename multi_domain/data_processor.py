import torch
import logging
import os


# 定义数据处理的父类
class NERProcessor:
    def __init__(self, split, max_len):
        self.split_list = split.split()  # args参数定义那里有split的选项。所以split_list=[',', '，', '.', '。', '!', '！', '?', '？']
        self.max_len = max_len

    def load_train_dataset(self, file_path, tokenizer, class_list):
        return None

    def load_multidomain_train_single_model_dataset(self, data_dir, domains, tokenizer, tags_list):
        return None

    def load_multidomain_train_dataset(self, data_dir, domains, tokenizer):
        return None

    def load_dev_dataset(self, file_path, tokenizer, class_list):
        return None, None

    def load_test_dataset(self, file_path, tokenizer, class_list):
        return None, None

    def get_tag_list(self):
        return None

    # 加载train，dev，test文本文件（格式为：字 标签），返回的sentences是[[[字，标签]，[字，标签]，[字，标签]...]，[[]...],[[]...],[[]...]...]
    def _load_data(self, file_path, mode=None):
        sentences = []
        with open(file_path, "r", encoding="utf8") as fin:
            sentence = []
            for line in fin.readlines():
                data = line.replace('\n', '').split('\t')
                if len(data) == 2:
                    sentence.append(data)
                else:
                    sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                sentences.append(sentence)
        if mode:
            logging.info(mode + ": Read \"" + file_path + "\" sentence number:" + str(len(sentences)))
        else:
            logging.info("Read \"" + file_path + "\" sentence number:" + str(len(sentences)))
        return sentences

    # 这是span方法用到的函数
    # 每个句子都对应一个start_labels,长度为句长，记录的是每个实体开始的位置。参数data就是一个句子的labels。
    # 每个句子都对应一个end_labels，长度为句长，记录的是每个实体最后的位置。
    def _bio_to_se(self, data):
        start_labels = ['O'] * len(data)
        end_labels = ['O'] * len(data)
        i = 0
        while i < len(data):
            if data[i][0] == 'B':
                tag = data[i][2:]
                start_labels[i] = tag
                i += 1
                while i < len(data) and data[i] == "I-" + tag:
                    i += 1
                i -= 1
                end_labels[i] = tag
            i += 1
        return start_labels, end_labels

    # 对句子按照最大句长截断句子。如果是span架构，则返回result。如果是crf或者softmax架构，则返回new_data。
    # error_num表示的是超过最大句长但是没有标点的句子数目，这样的就强行截断。
    def _split(self, data, label_flag=False, mode="crf"):
        result = []
        new_data = []
        original_num = len(data)
        error_num = 0
        for sentence in data:
            data_list = []
            new_sentence = sentence
            while len(new_sentence) >= self.max_len - 2:
                j = self.max_len - 3
                while j > 0:
                    if new_sentence[j][0] in self.split_list:
                        break
                    else:
                        j -= 1
                if j == 0:
                    j = self.max_len - 3
                    if label_flag:
                        while j + 1 >= len(new_sentence) or new_sentence[j + 1][1][0] == 'I':
                            j -= 1
                        if j == 0:
                            j = self.max_len - 3
                    error_num += 1
                data_list.append(new_sentence[:j + 1])
                new_sentence = new_sentence[j + 1:]
            if len(new_sentence) > 0:
                data_list.append(new_sentence)
            for data in data_list:
                text = []
                labels = []
                for char in data:
                    text.append(char[0])
                    labels.append(char[1])
                new_data.append((text, labels))
                if mode == "span":
                    start_labels, end_labels = self._bio_to_se(labels)
                    result.append((text, start_labels, end_labels))
        logging.info("final sentences: " + str(len(new_data)) + ", original sentences: " + str(original_num) +
                     ", error num:" + str(error_num))
        return result, new_data


# span架构所需的数据处理类
class SpanProcessor(NERProcessor):
    def load_train_dataset(self, train_file, tokenizer, class_list):
        data = self._load_data(train_file, "training")
        data, _ = self._split(data, True, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, "Traning", num=5)
        return datasets

    def load_multidomain_single_model_train_dataset(self, data_dir, domains, tokenizer, class_list):
        data_list = []
        for domain in domains:
            data = self._load_data(os.path.join(data_dir, domain, "train.txt"), "training")
            data_list.extend(data)
        data, _ = self._split(data_list, True, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, "Traning", num=5)
        return datasets

    def load_multidomain_train_dataset(self, data_dir, domain_tags, tokenizer):
        data_list = []
        domain_list = []
        for domain in domain_tags[:-1]:
            domain_list.append(domain[0])
            data = self._load_data(os.path.join(data_dir, domain[0], "train.txt"), "training")
            data, _ = self._split(data, True, mode="span")
            for sentence in data:
                data_list.append(sentence + (domain[0],))
        datasets = self._make_dataset(data_list, tokenizer, domain_tags[-1][1], "Traning", domain_list=domain_list, num=5)
        return datasets

    def load_dev_dataset(self, dev_file, tokenizer, class_list):
        data = self._load_data(dev_file, "development")
        data, new_data = self._split(data, True, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, "development", num=5)
        return datasets, new_data

    def load_test_dataset(self, test_file, tokenizer, class_list):
        original_data = self._load_data(test_file, "testing")
        data, _ = self._split(original_data, False, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, "Testing", num=5)
        return datasets, original_data

    # span架构，取出data下的class.txt文件标签，并返回class_list。
    def get_tags_list(self, tags_file, domain=None):
        class_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    class_list.append(data)
        if domain is not None:
            logging.info(domain + " class:" + str(class_list))
        else:
            logging.info("Class:" + str(class_list))
        return class_list

    def get_domain_tags(self, data_dir, domains):
        domain_tags = []
        all_list = []
        for domain in domains:
            tags_list = self.get_tags_list(os.path.join(data_dir, domain, "class.txt"), domain)
            domain_tags.append((domain, tags_list))
            for tag in tags_list:
                if tag not in all_list:
                    all_list.append(tag)
        domain_tags.append(('all', all_list))
        return domain_tags

    # 把train，dev，test文本转化成ID号。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, class_list, mode, domain_list=None, num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。
        def change_labels_to_ids(data, class_dict):
            result = [class_dict['O']]
            for label in data:
                result.append(class_dict[label])
            result.append(class_dict['O'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        start_labels_ids_list = []
        end_labels_ids_list = []
        domain_label_ids_list = []
        class_dict = {x: i + 1 for i, x in enumerate(class_list)}
        class_dict['O'] = 0
        for i, sentence in enumerate(data):
            text = sentence[0]
            start_labels = sentence[1]
            end_labels = sentence[2]
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            start_label_ids = change_labels_to_ids(start_labels, class_dict)
            end_label_ids = change_labels_to_ids(end_labels, class_dict)
            if domain_list is not None:
                domain_label_ids = domain_list.index(sentence[3])
            assert len(start_label_ids) == len(end_label_ids) == len(token_ids)

            while len(token_ids) < self.max_len:
                token_ids.append(0)
                mask_ids.append(0)
                start_label_ids.append(class_dict['O'])
                end_label_ids.append(class_dict['O'])
            assert len(token_ids) == self.max_len
            assert len(mask_ids) == self.max_len
            assert len(start_label_ids) == self.max_len
            assert len(end_label_ids) == self.max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("start_label_ids: %s" % " ".join([str(x) for x in start_label_ids]))
                logging.info("end_label_ids: %s" % " ".join([str(x) for x in end_label_ids]))
                if domain_list is not None:
                    logging.info("domain_label_ids: %s" % " ".join(str(domain_label_ids)))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            start_labels_ids_list.append(start_label_ids)
            end_labels_ids_list.append(end_label_ids)
            if domain_list is not None:
                domain_label_ids_list.append(domain_label_ids)

        if domain_list is not None:
            return torch.utils.data.TensorDataset(
                torch.tensor(token_ids_list, dtype=torch.long),
                torch.tensor(mask_ids_list, dtype=torch.long),
                torch.tensor(token_type_ids_list, dtype=torch.long),
                torch.tensor(start_labels_ids_list, dtype=torch.long),
                torch.tensor(end_labels_ids_list, dtype=torch.long),
                torch.tensor(domain_label_ids_list, dtype=torch.long)
            )
        else:
            return torch.utils.data.TensorDataset(
                torch.tensor(token_ids_list, dtype=torch.long),
                torch.tensor(mask_ids_list, dtype=torch.long),
                torch.tensor(token_type_ids_list, dtype=torch.long),
                torch.tensor(start_labels_ids_list, dtype=torch.long),
                torch.tensor(end_labels_ids_list, dtype=torch.long),
            )


# CRF架构所需的数据处理类
class CRFProcessor(NERProcessor):

    def load_train_dataset(self, train_file, tokenizer, tags_list):
        data = self._load_data(train_file, "training")
        _, data = self._split(data, True, mode="crf")
        datasets = self._make_dataset(data, tokenizer, tags_list, "Traning", 5)
        return datasets

    def load_multidomain_single_model_train_dataset(self, data_dir, domains, tokenizer, tags_list):
        data_list = []
        for domain in domains:
            data = self._load_data(os.path.join(data_dir, domain, "train.txt"), "training")
            data_list.extend(data)
        _, data = self._split(data_list, True, mode="crf")
        datasets = self._make_dataset(data, tokenizer, tags_list, "Traning", 5)
        return datasets

    def load_dev_dataset(self, dev_file, tokenizer, class_list):
        data = self._load_data(dev_file, "development")
        _, data = self._split(data, True, mode="crf")
        datasets = self._make_dataset(data, tokenizer, class_list, "development", 5)
        return datasets, data

    def load_test_dataset(self, test_file, tokenizer, class_list):
        original_data = self._load_data(test_file, "testing")
        _, data = self._split(original_data, False, mode="crf")
        datasets = self._make_dataset(data, tokenizer, class_list, "Testing", 5)
        return datasets, original_data

    # CRF架构，取出data下tags标签，并加入<s>,<e>,<p>标签，返回tags_list
    def get_tags_list(self, tags_file):
        tags_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    tags_list.append(data)
        tags_list.append('<s>')
        tags_list.append('<e>')
        tags_list.append('<p>')
        logging.info("Tags:" + str(tags_list))
        return tags_list

    # 把train，dev，test文本转化成ID号。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, tags_list, mode, num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。
        def change_labels_to_ids(data, tags_dict):
            result = [tags_dict['<s>']]
            for label in data:
                result.append(tags_dict[label])
            result.append(tags_dict['<e>'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        labels_ids_list = []
        tags_dict = {x: i for i, x in enumerate(tags_list)}
        for i, sentence in enumerate(data):
            text = sentence[0]
            labels = sentence[1]
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            label_ids = change_labels_to_ids(labels, tags_dict)
            assert len(label_ids) == len(token_ids)

            while len(token_ids) < self.max_len:
                token_ids.append(0)
                mask_ids.append(0)
                label_ids.append(tags_dict['<p>'])

            assert len(token_ids) == self.max_len
            assert len(mask_ids) == self.max_len
            assert len(label_ids) == self.max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            labels_ids_list.append(label_ids)

        return torch.utils.data.TensorDataset(
            torch.tensor(token_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(token_type_ids_list, dtype=torch.long),
            torch.tensor(labels_ids_list, dtype=torch.long)
        )


# Softmax架构所需的数据处理类，可以继承CRF的数据处理类
class SoftMaxProcessor(CRFProcessor):

    # 取出softmax架构下tags标签，返回tags_list。这里与CRF稍有不同，不会加入<s>、<p>、<e>这三个多的标签。
    def get_tags_list(self, tags_file):
        tags_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    tags_list.append(data)
        logging.info("Tags:" + str(tags_list))
        return tags_list

    # 把train，dev，test文本转化成ID。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, tags_list, mode="softmax", num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。不用加开始、填充和结束字符。
        def change_labels_to_ids(data, tags_dict):
            result = [tags_dict['O']]
            for label in data:
                result.append(tags_dict[label])
            result.append(tags_dict['O'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        labels_ids_list = []
        tags_dict = {x: i for i, x in enumerate(tags_list)}
        for i, sentence in enumerate(data):
            text = sentence[0]
            labels = sentence[1]
            # 经过tokenizer.encode()会添加上特殊字符[CLS]和[SEP]，所以label_ids要补充两个O字符
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            label_ids = change_labels_to_ids(labels, tags_dict)
            assert len(label_ids) == len(token_ids)

            while len(token_ids) < self.max_len:
                token_ids.append(0)
                mask_ids.append(0)
                label_ids.append(tags_dict['O'])

            assert len(token_ids) == self.max_len
            assert len(mask_ids) == self.max_len
            assert len(label_ids) == self.max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            labels_ids_list.append(label_ids)

        return torch.utils.data.TensorDataset(
            torch.tensor(token_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(token_type_ids_list, dtype=torch.long),
            torch.tensor(labels_ids_list, dtype=torch.long)
        )
