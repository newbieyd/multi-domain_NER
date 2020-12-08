import torch
import transformers
from torchcrf import CRF

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