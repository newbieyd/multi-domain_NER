import argparse
import logging
import os
import time

import torch
import transformers
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model import SpanModel, CRFModel, SoftmaxModel
from data_processor import SpanProcessor, CRFProcessor, SoftMaxProcessor
from utils import set_seed, calculate


# 评价过程，打印各类实体F1值和总体F1值
def evaluate(data, tags_list, title, mode, test_flag=False):
    def change_label_span(start_tags, end_tags, length):
        i = 0
        result = []
        while i < length:
            if start_tags[i] != 0:
                tag = start_tags[i]
                start_index = i
                while i < length and end_tags[i] == 0:
                    i += 1
                if i < length and end_tags[i] == tag:
                    result.append((start_index, i + 1, tag.item()))
            i += 1
        return result

    # CRF架构和softmax架构共用
    def change_label_bio(data, length, tags_list):
        i = 0
        result = []
        while i < length:
            if tags_list[data[i]][0] == 'B':
                tag = tags_list[data[i]][2:]
                start_index = i
                i += 1
                while i < length and tags_list[data[i]] == 'I-' + tag:
                    i += 1
                result.append((start_index, i, tag))
                i -= 1
            i += 1
        return result

    result_f1 = None
    result_dict = {}
    if mode == 'span':
        domain_entities_dict = {x + 1: [0, 0, 0] for x in range(len(tags_list))}
    elif mode == 'crf' or mode == "softmax":
        domain_entities_dict = {}
        for tag in tags_list:
            if tag[0] == 'B':
                domain_entities_dict[tag[2:]] = [0, 0, 0]
    domain_result_dict = {}
    logging.info("***** {} Evaluation *****".format(title))
    for domain, domain_data in data.items():
        outputs = domain_data['outputs']
        labels = domain_data['labels']
        mask_ids = domain_data["mask_ids"]
        sentence_num = outputs['num']
        if mode == 'span':
            entities_dict = {x + 1: [0, 0, 0] for x in range(len(tags_list))}
        elif mode == 'crf' or mode == "softmax":
            entities_dict = {}
            for tag in tags_list:
                if tag[0] == 'B':
                    entities_dict[tag[2:]] = [0, 0, 0]
        result_list = []
        for i in range(sentence_num):
            if mode == 'span':
                length = mask_ids[i].sum()
                predict_list = change_label_span(outputs['start_outputs'][i], outputs['end_outputs'][i], length)
            elif mode == 'softmax':
                # 注意解码长度的问题。crf是人家写好的，正好解码句长+2个标签。所以softmax方法应该和span方法一样，取length=mask_ids[i].sum()。不然会按最大句长解码。
                length = mask_ids[i].sum()
                predict_list = change_label_bio(outputs['outputs'][i], length, tags_list)
            elif mode == 'crf':
                length = len(outputs['outputs'][i])
                predict_list = change_label_bio(outputs['outputs'][i], length, tags_list)
            result_list.append((predict_list, length - 2))
            if not test_flag:
                if mode == 'span':
                    label_list = change_label_span(labels['start_labels_ids'][i], labels['end_labels_ids'][i], length)
                elif mode == 'softmax' or mode == 'crf':
                    label_list = change_label_bio(labels['labels_ids'][i], length, tags_list)
                for label in label_list:
                    entities_dict[label[2]][1] += 1
                for predict in predict_list:
                    entities_dict[predict[2]][0] += 1
                    if predict in label_list:
                        entities_dict[predict[2]][2] += 1
        result_dict[domain] = result_list
        if not test_flag:
            all_result = [0, 0, 0]
            for entity in entities_dict:
                for i in range(len(entities_dict[entity])):
                    all_result[i] += entities_dict[entity][i]
            logging.info("***** {} *****".format(domain))
            p, r, f1 = calculate(all_result)
            logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
                p, r, f1, all_result[0], all_result[1], all_result[2]))
            for tag_type in entities_dict:
                if mode == "span":
                    tag = tags_list[tag_type - 1]
                elif mode == 'softmax':
                    tag = tag_type
                elif mode == "crf":
                    tag = tag_type
                p, r, f1 = calculate(entities_dict[tag_type])
                logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                             "right: {}".format(tag, p, r, f1, entities_dict[tag_type][0],
                                                entities_dict[tag_type][1], entities_dict[tag_type][2]))
            for entity in entities_dict:
                for i in range(len(entities_dict[entity])):
                    domain_entities_dict[entity][i] += entities_dict[entity][i]
    if not test_flag:
        all_result = [0, 0, 0]
        for entity in domain_entities_dict:
            for i in range(len(domain_entities_dict[entity])):
                all_result[i] += domain_entities_dict[entity][i]
        logging.info("***** ALL *****")
        p, r, f1 = calculate(all_result)
        logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
            p, r, f1, all_result[0], all_result[1], all_result[2]))
        result_f1 = f1
        for tag_type in domain_entities_dict:
            if mode == "span":
                tag = tags_list[tag_type - 1]
            elif mode == 'softmax':
                tag = tag_type
            elif mode == "crf":
                tag = tag_type
            p, r, f1 = calculate(domain_entities_dict[tag_type])
            logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                         "right: {}".format(tag, p, r, f1, domain_entities_dict[tag_type][0],
                                            domain_entities_dict[tag_type][1], domain_entities_dict[tag_type][2]))
    return result_f1, result_dict


def get_one_domain_predict(dev_dataloader, model, device, title, mode):
    if mode == "span":
        start_labels_ids_list = []
        end_labels_ids_list = []
        start_output_list = []
        end_output_list = []
    elif mode == "softmax" or mode == "crf":
        labels_ids_list = []
        output_list = []
    mask_ids_list = []
    # tqdm进度条库，可视化
    for _, data in enumerate(tqdm(dev_dataloader, desc=title)):
        token_ids = data[0].to(device, dtype=torch.long)
        mask_ids = data[1].to(device, dtype=torch.long)
        token_type_ids = data[2].to(device, dtype=torch.long)
        if mode == "span":
            start_labels_ids = data[3].to(device, dtype=torch.long)
            end_labels_ids = data[4].to(device, dtype=torch.long)
            start_output, end_output = model(token_ids, mask_ids, token_type_ids)
            start_output = start_output.argmax(dim=-1)
            end_output = end_output.argmax(dim=-1)

            start_labels_ids_list.append(start_labels_ids)
            end_labels_ids_list.append(end_labels_ids)
            start_output_list.append(start_output)
            end_output_list.append(end_output)
        elif mode == "softmax":
            labels_ids = data[3].to(device, dtype=torch.long)
            output = model(token_ids, mask_ids, token_type_ids)
            output = output.argmax(dim=-1)

            output_list += output
            labels_ids_list.append(labels_ids)
        elif mode == "crf":
            labels_ids = data[3].to(device, dtype=torch.long)
            output = model(token_ids, mask_ids, token_type_ids)
            output = model.decode(output, mask_ids)

            labels_ids_list.append(labels_ids)
            output_list += output

        mask_ids_list.append(mask_ids)

    if mode == "span":
        start_labels_ids = torch.cat(start_labels_ids_list, dim=0)
        end_labels_ids = torch.cat(end_labels_ids_list, dim=0)
        start_outputs = torch.cat(start_output_list, dim=0)
        end_outputs = torch.cat(end_output_list, dim=0)
    elif mode == "crf" or mode == "softmax":
        labels_ids = torch.cat(labels_ids_list, dim=0)
    mask_ids_list = torch.cat(mask_ids_list, dim=0)

    outputs = {}
    labels = {}
    if mode == "span":
        outputs['start_outputs'] = start_outputs
        outputs['end_outputs'] = end_outputs
        outputs['num'] = start_labels_ids.size()[0]
        labels['start_labels_ids'] = start_labels_ids
        labels['end_labels_ids'] = end_labels_ids
    elif mode == "crf" or mode == "softmax":
        outputs['outputs'] = output_list
        outputs['num'] = len(output_list)
        labels['labels_ids'] = labels_ids
    return outputs, labels, mask_ids_list


# 如果有了模型，则进行验证，span，crf，softmax架构可选择
def development(model, device, dev_dataloader_dict, tags_list, mode):
    # 注意model.train()和model.eval()的不同作用。
    model.eval()
    start_time = time.time()
    all_data = {}
    sentences_num = 0
    for domain, dev_dataloader in dev_dataloader_dict.items():
        outputs, labels, mask_ids_list = get_one_domain_predict(dev_dataloader, model, device,
                                                                "Development-{}".format(domain), mode)
        sentences_num += len(dev_dataloader)
        all_data[domain] = {
            "outputs": outputs,
            "labels": labels,
            "mask_ids": mask_ids_list
        }

    f1, predict_dict = evaluate(all_data, tags_list, "Development", mode, False)

    end_time = time.time()
    logging.info("Development end, speed: {:.1f} sentences/s, all time: {:.2f}s".format(
        sentences_num / (end_time - start_time), end_time - start_time))

    return f1, predict_dict


# 训练过程
def train(args, model, device, train_datasets, dev_datasets, tags_list, writer):
    # 注意model.train()和model.eval()的不同作用。
    model.train()
    epoch_step = len(train_datasets) // args.train_batch_size + 1
    num_train_optimization_steps = epoch_step * args.epochs
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_datasets))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    # os.walk方法，主要用来遍历一个目录内各个子目录和子文件。
    # 可以得到一个三元tupple(dirpath, dirnames, filenames), 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    _, _, files = list(os.walk(args.output_dir))[0]
    epoch = 0
    for file in files:
        if len(file) > 0 and file[:10] == "checkpoint":
            temp = file[11:-4]
            if temp.isdigit() and int(temp) > epoch:
                epoch = int(temp)
    # 如果训练了几轮，保存了模型，那就直接导入模型。
    if epoch > 0:
        logging.info('checkpoint-' + str(epoch) + '.pkl is exit!')
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-' + str(epoch) + '.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-' + str(epoch) + '.pkl'))

    if epoch >= args.epochs:
        logging.info("The model has been trained!")
        return

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    if args.dev:
        dev_dataloader_dict = {}
        for domain, dev_dataset in dev_datasets.items():
            dev_dataloader = torch.utils.data.DataLoader(dev_dataset["dev_dataset"], batch_size=args.train_batch_size, shuffle=False)
            dev_dataloader_dict[domain] = dev_dataloader

    best_f1 = -1
    # 如果已经保存了最好的模型，就直接导入！
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        logging.info('checkpoint-best.pkl is exit!')
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        best_f1, _ = development(model, device, dev_dataloader_dict, tags_list, args.architecture)
        logging.info("Load best F1={:.4f}".format(best_f1))

    if args.architecture == "span" or "softmax":
        optimizer = transformers.AdamW(params=model.parameters(), lr=args.learning_rate)
    if args.architecture == "crf":
        optimizer = transformers.AdamW(
            params=[
                {'params': model.bert.parameters()},
                {'params': model.dence.parameters(), 'lr': args.crf_lr},
                {'params': model.crf.parameters(), 'lr': args.crf_lr}
            ],
            lr=args.learning_rate)
    # 学习率预热函数，使学习率线性增长，然后到某一schedule，在线性/指数降低
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_optimization_steps) * args.warmup_proportion,
        num_training_steps=num_train_optimization_steps
    )

    # 开始训练
    for current_epoch in range(epoch, args.epochs):
        model.train()
        all_loss = 0
        start_time = time.time()
        all_step = 0
        for step, data in enumerate(tqdm(train_dataloader, desc="Training")):
            token_ids = data[0].to(device, dtype=torch.long)
            mask_ids = data[1].to(device, dtype=torch.long)
            token_type_ids = data[2].to(device, dtype=torch.long)
            if args.architecture == "span":
                start_labels_ids = data[3].to(device, dtype=torch.long)
                end_labels_ids = data[4].to(device, dtype=torch.long)
                label = {
                    "start_labels_ids": start_labels_ids,
                    "end_labels_ids": end_labels_ids
                }
                start_output, end_output = model(token_ids, mask_ids, token_type_ids)
                output = {
                    "start_output": start_output,
                    "end_output": end_output
                }
                loss = model.loss(output, label, mask_ids)
            elif args.architecture == "softmax":
                labels_ids = data[3].to(device, dtype=torch.long)
                output = model(token_ids, mask_ids, token_type_ids)
                loss = model.loss(output, labels_ids, mask_ids)
            elif args.architecture == "crf":
                labels_ids = data[3].to(device, dtype=torch.long)
                output = model(token_ids, mask_ids, token_type_ids)
                loss = model.loss(output, labels_ids, mask_ids)

            if writer:
                writer.add_scalar('loss', loss, global_step=current_epoch * epoch_step + step + 1)
                writer.add_scalar('learning_rate',
                                  optimizer.state_dict()['param_groups'][0]['lr'],
                                  global_step=current_epoch * epoch_step + step + 1)
            all_loss += loss.item()

            optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            # optimizer的step为什么不能放在min-batch那个循环之外，还有optimizer.step和loss.backward的区别：
            # https://blog.csdn.net/xiaoxifei/article/details/87797935
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            all_step += 1

        # pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系。
        # torch.optim模块中的Optimizer优化器对象也存在一个state_dict对象，此处的state_dict字典对象包含state和param_groups的字典对象，
        # 而param_groups key对应的value也是一个由学习率，动量等参数组成的一个字典对象。
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_time = time.time()
        logging.info("Epoch: {}, Loss: {:.3g}, learning rate: {:.3g}, Time: {:.2f}s".format(
            current_epoch + 1, all_loss / all_step, lr, end_time - start_time))
        torch.save(model, os.path.join(args.output_dir, 'checkpoint-' + str(current_epoch + 1) + '.pkl'))
        delet_checkpoints_name = os.path.join(args.output_dir, 'checkpoint-' + str(
            current_epoch + 1 - args.keep_last_n_checkpoints) + '.pkl')
        if os.path.exists(delet_checkpoints_name):
            os.remove(delet_checkpoints_name)
        if args.dev:
            f1, _ = development(model, device, dev_dataloader_dict, tags_list, args.architecture)
            if f1 == -1 or f1 > best_f1:
                best_f1 = f1
                logging.info("Best F1={:.4f}, save model!".format(best_f1))
                torch.save(model, os.path.join(args.output_dir, 'checkpoint-best.pkl'))
            if writer:
                writer.add_scalar('dev_f1', f1, global_step=current_epoch * epoch_step)
                writer.add_scalar('dev_best_f1', best_f1, global_step=current_epoch * epoch_step)

    torch.save(model, os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    if args.dev:
        f1, _ = development(model, device, dev_dataloader_dict, tags_list, args.architecture)
        if f1 == -1 or f1 > best_f1:
            best_f1 = f1
            logging.info("Best F1={:.4f}, save model!".format(best_f1))
            torch.save(model, os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    logging.info("Training end!")


def write_one_domain_to_file(file_path, sentences, predict_list, tags_list, mode):
    assert len(sentences) == len(predict_list)
    write_data_list = []
    for i in range(len(sentences)):
        sentence, label = sentences[i]
        predict, _ = predict_list[i]
        result = ['O'] * len(sentence)
        for entity in predict:
            if mode == 'span':
                tag = tags_list[entity[2] - 1]
            elif mode == 'softmax' or mode == 'crf':
                tag = entity[2]
            result[entity[0] - 1] = "B-" + tag
            for j in range(entity[0], entity[1] - 1):
                result[j] = "I-" + tag
        write_data = []
        for j in range(len(sentence)):
            write_data.append((sentence[j], label[j], result[j]))
        write_data_list.append(write_data)

    with open(file_path, "w", encoding="utf8") as fout:
        for sentence in write_data_list:
            for data in sentence:
                fout.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\n')
            fout.write('\n')


# 验证过程，直接加载模型
def dev(args, datasets, model, device, tags_list):
    # 加载模型，没有模型则报错
    if args.model is not None:
        model = torch.load(args.model)
        logging.info("Load model:" + args.model)
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-last.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-last.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    else:
        logging.info("Error! The model file does not exist!")
        exit(1)
    model.eval()
    dataloader_dict = {}
    for domain, dev_dataset in datasets.items():
        dataloader = torch.utils.data.DataLoader(dev_dataset["dev_dataset"], batch_size=args.train_batch_size,
                                                     shuffle=False)
        dataloader_dict[domain] = dataloader
    _, predict_dict = development(model, device, dataloader_dict, tags_list, args.architecture)
    dev_dir = os.path.join(args.output_dir, "development")
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)
    for domain in datasets:
        sentences = datasets[domain]['dev_data']
        predict_list = predict_dict[domain]
        assert len(sentences) == len(predict_list)
        write_data_list = []
        for i in range(len(sentences)):
            sentence, label = sentences[i]
            predict, _ = predict_list[i]
            result = ['O'] * len(sentence)
            for entity in predict:
                if args.architecture == 'span':
                    tag = tags_list[entity[2] - 1]
                elif args.architecture == 'softmax' or args.architecture == 'crf':
                    tag = entity[2]
                result[entity[0] - 1] = "B-" + tag
                for j in range(entity[0], entity[1] - 1):
                    result[j] = "I-" + tag
            write_data = []
            for j in range(len(sentence)):
                write_data.append((sentence[j], label[j], result[j]))
            write_data_list.append(write_data)

        with open(os.path.join(args.output_dir, "development", "{}.txt".format(domain)), "w", encoding="utf8") as fout:
            for sentence in write_data_list:
                for data in sentence:
                    fout.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\n')
                fout.write('\n')
    logging.info("Development data is written to directory: " + os.path.join(args.output_dir, "development") + '!')


# 测试过程，直接加载模型
def test(args, processor, tokenizer, model, device, domains, tags_list):
    if args.model is not None:
        model = torch.load(args.model)
        logging.info("Load model:" + args.model)
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-last.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-last.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    else:
        logging.info("Error! The model file does not exist!")
        exit(1)

    start_time = time.time()
    datasets = {}
    all_sentence_num = 0
    for domain in domains:
        logging.info("--- {} ---".format(domain))
        dataset, sentences = processor.load_test_dataset(os.path.join(args.data_dir, domain, "test.txt"), tokenizer,
                                                           tags_list)
        all_sentence_num += len(dataset)
        datasets[domain] = {
            "dataset": dataset,
            "sentences": sentences
        }
    model.eval()
    logging.info("***** Running Testing *****")
    logging.info("  Num examples = %d", all_sentence_num)
    logging.info("  Batch size = %d", args.test_batch_size)

    dataloader_dict = {}
    for domain, dataset in datasets.items():
        dataloader = torch.utils.data.DataLoader(dataset["dataset"], batch_size=args.test_batch_size, shuffle=False)
        dataloader_dict[domain] = dataloader
    all_data = {}
    for domain, dataloader in dataloader_dict.items():
        outputs, labels, mask_ids_list = get_one_domain_predict(dataloader, model, device,
                                                                "Test-{}".format(domain), args.architecture)
        all_data[domain] = {
            "outputs": outputs,
            "labels": labels,
            "mask_ids": mask_ids_list
        }
    _, predict_dict = evaluate(all_data, tags_list, "Testing", args.architecture, True)
    test_dir = os.path.join(args.output_dir, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    all_data = {}
    for domain in datasets:
        sentences = datasets[domain]['sentences']
        predict_list = predict_dict[domain]
        write_data_list = []
        m = 0
        for i in range(len(sentences)):
            sentence = sentences[i]
            result = []
            while len(result) < len(sentence):
                predict, length = predict_list[m]
                m += 1
                predict_tags = ['O'] * length
                for entity in predict:
                    if args.architecture == 'span':
                        tag = tags_list[entity[2] - 1]
                    elif args.architecture == 'softmax' or args.architecture == 'crf':
                        tag = entity[2]
                    predict_tags[entity[0] - 1] = "B-" + tag
                    for j in range(entity[0], entity[1] - 1):
                        predict_tags[j] = "I-" + tag
                result += predict_tags
            assert len(result) == len(sentence)
            write_data = []
            for j in range(len(sentence)):
                write_data.append((sentence[j][0], sentence[j][1], result[j]))
            write_data_list.append(write_data)
        all_data[domain] = write_data_list

        with open(os.path.join(args.output_dir, "test", "{}.txt".format(domain)), "w", encoding="utf8") as fout:
            for sentence in write_data_list:
                for data in sentence:
                    fout.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\n')
                fout.write('\n')
    logging.info("Prediction data is written to directory: " + os.path.join(args.output_dir, "test") + '!')

    end_time = time.time()
    logging.info("Testing end, speed: {:.1f} sentences/s, all time: {:.2f}s".format(
        all_sentence_num / (end_time - start_time), end_time - start_time))

    def change_label(tags):
        i = 0
        result = []
        while i < len(tags):
            if tags[i][0] == "B":
                class_type = tags[i][2:]
                start_index = i
                i += 1
                while i < len(tags) and tags[i] == "I-" + class_type:
                    i += 1
                result.append((start_index, i, class_type))
                i -= 1
            i += 1
        return result
    if args.architecture == 'span':
        domain_entities_dict = {x: [0, 0, 0] for x in tags_list}
    elif args.architecture == 'crf' or args.architecture == "softmax":
        domain_entities_dict = {}
        for tag in tags_list:
            if tag[0] == 'B':
                domain_entities_dict[tag[2:]] = [0, 0, 0]
    for domain, write_data_list in all_data.items():
        if args.architecture == "span":
            entities_dict = {x: [0, 0, 0] for x in tags_list}
        else:
            entities_dict = {}
            for tag in tags_list:
                if tag[0] == 'B':
                    entities_dict[tag[2:]] = [0, 0, 0]
        for sentence in write_data_list:
            predict_data = []
            label_data = []
            for data in sentence:
                predict_data.append(data[2])
                label_data.append(data[1])
            predict_list = change_label(predict_data)
            label_list = change_label(label_data)
            for label in label_list:
                entities_dict[label[2]][1] += 1
            for predict in predict_list:
                entities_dict[predict[2]][0] += 1
                if predict in label_list:
                    entities_dict[predict[2]][2] += 1
        all_result = [0, 0, 0]
        for entity in entities_dict:
            if entities_dict[entity][1] > 0:
                for i in range(len(entities_dict[entity])):
                    all_result[i] += entities_dict[entity][i]
        logging.info("***** {} *****".format(domain))
        p, r, f1 = calculate(all_result)
        logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
            p, r, f1, all_result[0], all_result[1], all_result[2]))
        for tag_type in entities_dict:
            if entities_dict[tag_type][1] > 0:
                p, r, f1 = calculate(entities_dict[tag_type])
                logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                             "right: {}".format(tag_type, p, r, f1, entities_dict[tag_type][0],
                                                entities_dict[tag_type][1], entities_dict[tag_type][2]))
        for entity in entities_dict:
            if entities_dict[entity][1] > 0:
                for i in range(len(entities_dict[entity])):
                    domain_entities_dict[entity][i] += entities_dict[entity][i]
    all_result = [0, 0, 0]
    for entity in domain_entities_dict:
        for i in range(len(domain_entities_dict[entity])):
            all_result[i] += domain_entities_dict[entity][i]
    logging.info("***** ALL *****")
    p, r, f1 = calculate(all_result)
    logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
        p, r, f1, all_result[0], all_result[1], all_result[2]))
    for tag_type in domain_entities_dict:
        p, r, f1 = calculate(domain_entities_dict[tag_type])
        logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                     "right: {}".format(tag_type, p, r, f1, domain_entities_dict[tag_type][0],
                                        domain_entities_dict[tag_type][1], domain_entities_dict[tag_type][2]))


def main():
    parser = argparse.ArgumentParser(description="Named Entity Recognition: Multi_domain Single Model")

    parser.add_argument("--data_dir", required=True, help="The data folder path.")
    parser.add_argument("--domain", required=True, help="The domain names (multiple domains separated by *)")
    parser.add_argument("--train", default=False, action='store_true', help="Training.")
    parser.add_argument("--dev", default=False, action='store_true', help="Development.")
    parser.add_argument("--test", default=False, action='store_true', help="Testing.")
    parser.add_argument("--tags_file", required=True, help="The tags file path.")

    parser.add_argument("--output_dir", required=True, help="The output folder path.")

    parser.add_argument("--model", default=None, help="The model path.")

    parser.add_argument("--architecture", default="span", choices=['span', 'crf', 'softmax'],
                        help="The model architecture of neural network and what decoding method is adopted.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="The number of sentences contained in a batch during training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="The number of sentences contained in a batch during testing.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_lr", default=0.0001, type=float,
                        help="The initial learning rate of CRF layer.")
    parser.add_argument("--max_len", required=True, type=int, help="The Maximum length of a sentence.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="What percentage of neurons are discarded in the fully connected layers (0 ~ 1).")
    parser.add_argument("--keep_last_n_checkpoints", default=1, type=int,
                        help="Keep the last n checkpoints.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--split", default=", . ， 。 ！ ？ ! ?", help="Characters that segments a sentence.")
    parser.add_argument("--tensorboard_dir", default=None, help="The data address of the tensorboard.")

    parser.add_argument("--bert_config_file", required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--cpu", default=False, action='store_true',
                        help="Whether to use CPU, if not and CUDA is avaliable can use CPU.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization.")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(funcName)s: %(message)s',
        datefmt='%m-%d-%Y-%H:%M:%S',
        filemode='w',
        level=logging.INFO
    )

    setting = vars(args)
    logging.info("-" * 20 + "args" + "-" * 20)
    for key, value in setting.items():
        logging.info('%-30s%-s' % (key, str(value)))

    # Set seed
    set_seed(args.seed)

    if not args.train and not args.dev and not args.test:
        raise ValueError("At least one of `train_file`, `dev_file` or `test_file` must be not None.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.info("Output directory: " + args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info("device: " + str(device))

    architecture = args.architecture

    tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_config_file)
    config = transformers.BertConfig(args.bert_config_file)

    processor = None
    tags_list = None
    model = None
    if architecture == "span":
        processor = SpanProcessor(args.split, args.max_len)
        tags_list = processor.get_tags_list(args.tags_file)
        model = SpanModel(args.bert_config_file, config, len(tags_list) + 1, args.dropout)
    elif architecture == 'softmax':
        processor = SoftMaxProcessor(args.split, args.max_len)
        tags_list = processor.get_tags_list(args.tags_file)
        model = SoftmaxModel(args.bert_config_file, config, len(tags_list), args.dropout)
    elif architecture == "crf":
        processor = CRFProcessor(args.split, args.max_len)
        tags_list = processor.get_tags_list(args.tags_file)
        model = CRFModel(args.bert_config_file, config, len(tags_list), args.dropout)

    model.to(device)
    logging.info(model)

    writer = None
    if args.tensorboard_dir:
        writer = SummaryWriter(args.tensorboard_dir)
        # writer.add_graph(model, (torch.zeros(1, 10).to(device).long(),
        #                          torch.zeros(1, 10).to(device).long(),
        #                          torch.zeros(1, 10).to(device).long()))

    dev_datasets = {}
    # 只使用 --train_file 则只训练到固定轮数，保存为最后的模型 checkpoint-last.kpl
    domains = args.domain.split('*')
    if args.train:
        train_datasets = processor.load_multidomain_train_dataset(args.data_dir, domains, tokenizer, tags_list)
    # 若使用 --train_file 和 --dev_file 则会额外域保存在开发集上的最高分数的模型 checkpoint-best.kpl
    if args.dev:
        for domain in domains:
            logging.info("--- {} ---".format(domain))
            dev_dataset, dev_data = processor.load_dev_dataset(os.path.join(args.data_dir, domain, "dev.txt"), tokenizer, tags_list)
            dev_datasets[domain] = {
                "dev_dataset": dev_dataset,
                "dev_data": dev_data
            }
    if args.train:
        train(args, model, device, train_datasets, dev_datasets, tags_list, writer)
    if args.dev:
        dev(args, dev_datasets, model, device, tags_list)
    if args.test:
        test(args, processor, tokenizer, model, device, domains, tags_list)


if __name__ == "__main__":
    main()
