import torch
import transformers

# NER模型的父类
class NERModel(torch.nn.Module):
    def __init__(self, bert_file_path):
        super(NERModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_file_path)

    def forward(self, ids, mask, token_type_ids):
        return None

    def loss(self, output, label, mask_ids):
        return None


# span架构模型
class SpanModel(NERModel):
    def __init__(self, bert_file_path, config, class_num, dropout):
        super(SpanModel, self).__init__(bert_file_path)
        self.start_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, class_num),
            torch.nn.ReLU()
        )
        self.end_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, class_num),
            torch.nn.ReLU()
        )

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        start_output = self.start_classifier(output)
        end_output = self.end_classifier(output)
        return start_output, end_output

    def loss(self, output, label, mask_ids):
        def calculate_loss(output, labels, mask_ids):
            bz, length = labels.shape
            mask = mask_ids.view(-1) == 1
            output = output.view(bz * length, -1)[mask]
            labels = labels.view(-1)[mask]
            return torch.nn.CrossEntropyLoss()(output, labels)
        start_output = output['start_output']
        start_labels_ids = label['start_labels_ids']
        end_output = output['end_output']
        end_labels_ids = label['end_labels_ids']
        loss_1 = calculate_loss(start_output, start_labels_ids, mask_ids)
        loss_2 = calculate_loss(end_output, end_labels_ids, mask_ids)
        return loss_1 + loss_2


class SpanDomainModel(NERModel):
    def __init__(self, bert_file_path, config, domain_tags, dropout, domain_loss_rate, domain_ner_loss_rate):
        super(SpanDomainModel, self).__init__(bert_file_path)
        self.domain_classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(config.hidden_size, len(domain_tags) - 1),
                torch.nn.Softmax(dim=-1)
            )
        self.start_classifier_list = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(config.hidden_size, len(domain_tags[-1][1]) + 1),
                torch.nn.ReLU()
            )
            for _ in range(len(domain_tags) - 1)])

        self.end_classifier_list = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(config.hidden_size, len(domain_tags[-1][1]) + 1),
                torch.nn.ReLU()
            )
            for _ in range(len(domain_tags) - 1)])
        self.domain_loss_rate = domain_loss_rate
        self.domain_ner_loss_rate = domain_ner_loss_rate

    def forward(self, ids, mask, token_type_ids):
        output, setence_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        start_output_list = []
        for classifier in self.start_classifier_list:
            start_output_list.append(classifier(output).unsqueeze(0))
        end_output_list = []
        for classifier in self.end_classifier_list:
            end_output_list.append(classifier(output).unsqueeze(0))
        start_output = torch.cat(start_output_list, dim=0)
        end_output = torch.cat(end_output_list, dim=0)
        domain_output = self.domain_classifier(setence_output)
        weight = domain_output.expand(start_output.size(-2)*start_output.size(-1), domain_output.size(0), domain_output.size(1))
        weight = weight.transpose(0, 2)
        weight = weight.reshape(domain_output.size(1), domain_output.size(0), start_output.size(-2), start_output.size(-1))
        final_start_output = torch.sum(start_output * weight, dim=0)
        final_end_output = torch.sum(end_output * weight, dim=0)
        return {
            "start_output_list": start_output_list,
            "end_output_list": end_output_list,
            "final_start_output": final_start_output,
            "final_end_output": final_end_output,
            "domain_output": domain_output
        }

    def loss(self, output, label, mask_ids):
        def calculate_sentence_loss(output, labels, mask_ids):
            loss_list = []
            for i in range(output.size(1)):
                sentence_output = output[0][i][mask_ids[i]]
                sentence_labels = labels[i][mask_ids[i]]
                sentencec_loss = torch.nn.CrossEntropyLoss()(sentence_output, sentence_labels)
                loss_list.append(sentencec_loss.unsqueeze(0))
            return torch.cat(loss_list)
        def calculate_batch_loss(output, labels, mask_ids):
            bz, length = labels.shape
            mask = mask_ids.view(-1) == 1
            output = output.view(bz * length, -1)[mask]
            labels = labels.view(-1)[mask]
            return torch.nn.CrossEntropyLoss()(output, labels)
        loss_list = []
        for i in range(len(output['start_output_list'])):
            start_output = output['start_output_list'][i]
            end_output = output['end_output_list'][i]
            start_loss = calculate_sentence_loss(start_output, label['start_labels_ids'], mask_ids)
            end_loss = calculate_sentence_loss(end_output, label['end_labels_ids'], mask_ids)
            sentence_loss = start_loss + end_loss
            loss_list.append(sentence_loss.unsqueeze(0))
        sentence_loss = torch.cat(loss_list, dim=0).transpose(0, 1) * output["domain_output"]
        sentence_loss = torch.mean(sentence_loss)
        domain_loss = torch.nn.CrossEntropyLoss()(output["domain_output"], label["domain_labels_ids"])
        final_ner_start_loss = calculate_batch_loss(output["final_start_output"], label["start_labels_ids"], mask_ids)
        final_ner_end_loss = calculate_batch_loss(output["final_end_output"], label["end_labels_ids"], mask_ids)
        final_ner_loss = final_ner_start_loss + final_ner_end_loss
        loss = self.domain_ner_loss_rate * sentence_loss + self.domain_loss_rate * domain_loss + final_ner_loss
        return loss


# Softmax架构模型！注意在pytorch中，交叉熵函数，会自动加一层softmax激活函数
# 所以，训练时不需要自己把网络的输出结果再经过一次softmax了。解码的时候，也是选最大的，所以应该也不用softmax了。
class SoftmaxModel(NERModel):
    # 这个dropout和linear的顺序，应该影响不大
    def __init__(self, bert_file_path, config, tags_num, dropout):
        super(SoftmaxModel, self).__init__(bert_file_path)
        self.dence = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, tags_num),
            torch.nn.ReLU()
        )

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.dence(output)
        return output

    def loss(self, output, labels_ids, mask_ids):
        bz, length = labels_ids.shape
        mask = mask_ids.view(-1) == 1
        output = output.view(bz * length, -1)[mask]
        labels = labels_ids.view(-1)[mask]
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        return loss


# CRF架构模型
class CRFModel(NERModel):
    def __init__(self, bert_file_path, config, tags_num, dropout):
        super(CRFModel, self).__init__(bert_file_path)
        self.dence = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, tags_num),
            torch.nn.ReLU()
        )
        self.crf = CRF(tags_num, batch_first=True)

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.dence(output)
        return output

    def loss(self, output, labels_ids, mask_ids):
        mask = mask_ids == 1
        return -self.crf(output, labels_ids, mask=mask, reduction='mean')

    def decode(self, output, mask_ids):
        mask = mask_ids == 1
        return self.crf.decode(output, mask)
