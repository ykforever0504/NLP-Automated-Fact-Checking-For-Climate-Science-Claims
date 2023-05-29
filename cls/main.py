import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from transformers import AutoTokenizer, AutoModel
from data_utils import to_cuda, ClsDataset
from torch.utils.data import DataLoader
from config import cls_setting
from tqdm import tqdm
from model import CLSModel
import json
import wandb

wandb.init(project="nlp", name="cls")

def train(args):
    cls_setting(args)

    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_type)

    label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}

    train_set = ClsDataset("train", label2ids, tok, args.max_length)
    val_set = ClsDataset("dev", label2ids, tok, args.max_length)

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn)

   # build models
    model = CLSModel(args.model_type)

    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))
    model.cuda()
    model.train()

    cache_name = "cls_k"
    save_dir = f"./cache/{cache_name}"
    os.makedirs(save_dir, exist_ok=True)

    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.3)
    s_optimizer = optim.Adam(model.parameters())

    for param in s_optimizer.param_groups:
        param['lr'] = args.max_lr

    # start training
    s_optimizer.zero_grad()
    step_cnt = 0
    all_step_cnt = 0
    avg_loss = 0
    maximum_acc = 0

    for epoch in range(args.epoch):
        epoch_step = 0
        for (i, batch) in enumerate(tqdm(dataloader)):
            to_cuda(batch)
            step_cnt += 1
            # forward pass
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            loss = ce_fn(logits, batch["label"])
            loss = loss / args.accumulate_step
            loss.backward()

            avg_loss += loss.item()
            if step_cnt == args.accumulate_step:
                # back
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                if all_step_cnt <= args.lr_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.lr_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.lr_steps) * 4e-9

                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()

            if all_step_cnt % args.report_freqence == 0 and step_cnt == 0:
                if all_step_cnt <= args.lr_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.lr_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.lr_steps) * 4e-9

                wandb.log({"learning_rate": lr}, step=all_step_cnt)
                wandb.log({"loss": avg_loss / args.report_freqence}, step=all_step_cnt)
                # report stats
                print("\n")
                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_step, avg_loss / args.report_freqence))
                print(f"learning rate: {lr:.6f}")
                print("\n")

                avg_loss = 0
            del loss, logits

            if all_step_cnt % args.val_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("\nEvaluate:\n")
                acc = validate(val_dataloader, model)
                wandb.log({"accuracy": acc}, step=all_step_cnt)

                if acc > maximum_acc:
                    maximum_acc = acc
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_ckpt.bin"))
                    print("\n")
                    print("best val loss - epoch: %d, epoch_step: %d" % (epoch, epoch_step))
                    print("maximum_acc", acc)
                    print("\n")

                    
def validate(val_dataloader, model):
    model.eval()
    cnt = 0.
    correct_cnt = 0.
    for batch in tqdm(val_dataloader):
        to_cuda(batch)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1)
        result = predict_labels == batch["label"]
        correct_cnt += result.sum().item()
        cnt += predict_labels.size(0)
    acc = correct_cnt / cnt
    print("\n")
    print("evaluation accuracy: %.3f" % acc)
    print("\n")

    model.train()

    return acc


def predict(args):
    # load data
    cls_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    label2ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
    ids2label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    test_set = ClsDataset("test", label2ids, tok, args.max_length)
    dev_set = ClsDataset("dev", label2ids, tok, args.max_length)

    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate_fn)
    # build models
    model = CLSModel(args.model_type)

    assert len(args.model_pt) > 0
    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))

    model.cuda()
    model.eval()

    out_data = {}
    for batch in tqdm(test_dataloader):
        to_cuda(batch)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1).tolist()
        idx = 0
        for data, predict_label in zip(batch["datas"], predict_labels):
            data["claim_label"] = ids2label[predict_label]
            out_data[batch["claim_ids"][idx]] = data
            idx += 1
    fout = open("test-claims-predictions.json", 'w')
    json.dump(out_data, fout)
    fout.close()
    
    out_data = {}
    for batch in tqdm(dev_dataloader):
        to_cuda(batch)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predict_labels = logits.argmax(-1).tolist()
        idx = 0
        for data, predict_label in zip(batch["datas"], predict_labels):
            data["claim_label"] = ids2label[predict_label]
            out_data[batch["claim_ids"][idx]] = data
            idx += 1
    fout = open("dev-claims-predictions.json", 'w')
    json.dump(out_data, fout)
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("-p", "--predict", action="store_true", help="prediction")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    args = parser.parse_args()

    if args.predict:
        predict(args)
    else:
        train(args)


