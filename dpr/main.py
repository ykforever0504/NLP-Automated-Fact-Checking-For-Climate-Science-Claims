import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
import json
import wandb
from transformers import AutoTokenizer, AutoModel
from data_utils import to_cuda, TrainDataset, ValDataset, EvidenceDataset
from torch.utils.data import DataLoader
from config import dpr_setting
from tqdm import tqdm

wandb.init(project="nlp", name="dpr")

def train(args):
    dpr_setting(args)

    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # download model from the cloud
    tok = AutoTokenizer.from_pretrained(args.model_type)
    # load dataset
    train_set = TrainDataset("train", tok, args.evidence_samples_num, args.max_length)
    dev_set = ValDataset("dev", tok, args.max_length)
    evidence_set = EvidenceDataset(tok, args.max_length)

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    val_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate_fn)
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)

    # build models
    encoder_model = AutoModel.from_pretrained(args.model_type)

    # query_model = AutoModel.from_pretrained(args.model_type)
    # evidence_model = AutoModel.from_pretrained(args.model_type)
    
    #if there is any existing model, then continue the last model
    if len(args.model_pt) > 0:
        encoder_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))
        # query_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "query_ckpt.bin")))
        # evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "evidence_ckpt.bin")))

    encoder_model.cuda()
    # query_model.cuda()
    # evidence_model.cuda()
    # query_model.eval()
    # evidence_model.eval()
    
    #save model to the cache
    cache_name = "dpr"
    save_dir = f"./cache/{cache_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    #using the adam optimizer
    encoder_optimizer = optim.Adam(encoder_model.parameters())
    # query_optimizer = optim.Adam(query_model.parameters())
    # evidence_optimizer = optim.Adam(evidence_model.parameters())

    # lr 
    for param in encoder_optimizer.param_groups:
        param['lr'] = args.max_lr
    # for param in query_optimizer.param_groups:
    #     param['lr'] = args.max_lr
    # for param in evidence_optimizer.param_groups:
    #     param['lr'] = args.max_lr

    # start training
    encoder_optimizer.zero_grad()
    # query_optimizer.zero_grad()
    # evidence_optimizer.zero_grad()

    step_cnt = 0
    all_step_cnt = 0
    avg_loss = 0
    maximum_f_score = 0

    print("\nEvaluate:\n")
    # evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, query_model, evidence_model)
    # f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, query_model, evidence_model)
    evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, encoder_model)
    f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, encoder_model)
    wandb.log({"f_score": f_score}, step=all_step_cnt)

    for epoch in range(args.epoch): # Loop over the specified number of epochs
        epoch_step = 0 # Set epoch_step to 0 before starting the epoch

        for (i, batch) in enumerate(tqdm(dataloader)):  # Loop over batches in the dataloader
            to_cuda(batch) # Move the batch to GPU
            step_cnt += 1 # Increment the step counter
            # Compute the query and evidence embeddings
            # query_embeddings = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
            # evidence_embeddings = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
            query_embeddings = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
            evidence_embeddings = encoder_model(input_ids=batch["evidence_input_ids"],attention_mask=batch["evidence_attention_mask"]).last_hidden_state
            query_embeddings = query_embeddings[:, 0, :]
            evidence_embeddings = evidence_embeddings[:, 0, :]
            
            # Normalize the embeddings
            query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=1)
            evidence_embeddings = nn.functional.normalize(evidence_embeddings, p=2, dim=1)
            # Compute the similarity and log softmax loss
            sims = torch.mm(query_embeddings, evidence_embeddings.t())
            scores = - nn.functional.log_softmax(sims / 0.05, dim=1)
            ###
            
            # Compute the loss for each label in the batch
            loss = []
            start_idx = 0
            for idx, label in enumerate(batch["labels"]):
                end_idx = start_idx + label
                cur_loss = torch.mean(scores[idx, start_idx:end_idx])
                loss.append(cur_loss)
                start_idx = end_idx
            loss = torch.stack(loss).mean() # Compute the mean loss for the batch
            loss = loss / args.accumulate_step # Divide the loss by the number of steps to accumulate before updating the model
            loss.backward() # Backpropagate the loss
            avg_loss += loss.item()  # Add the loss to the running total
            if step_cnt == args.accumulate_step: # If the step counter has reached the specified number of steps to accumulate, update the model
               # Clip the gradients if the gradient norm threshold is greater than 0
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(encoder_model.parameters(), args.grad_norm)
                    # nn.utils.clip_grad_norm_(query_model.parameters(), args.grad_norm)
                    # nn.utils.clip_grad_norm_(evidence_model.parameters(), args.grad_norm)

                step_cnt = 0
                # Increment the epoch step counter and the total step counter
                epoch_step += 1
                all_step_cnt += 1
                # Adjust the learning rate
                if all_step_cnt <= args.lr_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.lr_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.lr_steps) * 1e-8
                # Update the optimizer learning rate
                for param in encoder_optimizer.param_groups:
                    param['lr'] = lr
                encoder_optimizer.step()
                encoder_optimizer.zero_grad()
#                 for param in query_optimizer.param_groups:
#                     param['lr'] = lr
#                 for param in evidence_optimizer.param_groups:
#                     param['lr'] = lr

#                 query_optimizer.step()
#                 evidence_optimizer.step()
#                 query_optimizer.zero_grad()
#                 evidence_optimizer.zero_grad()
                
            # Log the learning rate and average loss to Weights and Biases
            if all_step_cnt % args.report_freqence == 0 and step_cnt == 0:
                if all_step_cnt <= args.lr_steps:
                    lr = all_step_cnt * (args.max_lr - 2e-8) / args.lr_steps + 2e-8
                else:
                    lr = args.max_lr - (all_step_cnt - args.lr_steps) * 1e-8

                wandb.log({"learning_rate": lr}, step=all_step_cnt)
                wandb.log({"loss": avg_loss / args.report_freqence}, step=all_step_cnt)
                # Print the epoch, epoch step, and average loss
                print("\n")
                print("epoch: %d, epoch_step: %d, avg loss: %.6f" % (epoch + 1, epoch_step, avg_loss / args.report_freqence))
                print(f"learning rate: {lr:.6f}")
                print("\n")

                avg_loss = 0
            del loss, cos_sims, query_embeddings, evidence_embeddings

            if all_step_cnt % args.val_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                print("\nEvaluate:\n")
                # evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, query_model, evidence_model)
                # f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, query_model, evidence_model)
                evidence_embeddings, evidence_ids = get_evidence_embeddings(evidence_dataloader, encoder_model)
                f_score = validate(val_dataloader, evidence_embeddings, evidence_ids, encoder_model)
                wandb.log({"f_score": f_score}, step=all_step_cnt)

                if f_score > maximum_f_score:
                    maximum_f_score = f_score
                    torch.save(encoder_model.state_dict(), os.path.join(save_dir, "best_ckpt.bin"))
                    # torch.save(query_model.state_dict(), os.path.join(save_dir, "query_ckpt.bin"))
                    # torch.save(evidence_model.state_dict(), os.path.join(save_dir, "evidence_ckpt.bin"))
                    print("\n")
                    print("best val loss - epoch: %d, epoch_step: %d" % (epoch, epoch_step))
                    print("maximum_f_score", f_score)
                    print("\n")

# def validate(val_dataloader, evidence_embeddings, evidence_ids, query_model, evidence_model):
def validate(val_dataloader, evidence_embeddings, evidence_ids, encoder_model):
    # initialize empty list to store f-scores for each query
    fscores = []
    # iterate through validation dataloader
    for batch in tqdm(val_dataloader):
        to_cuda(batch)
        # get query embedding
        query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        # query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        # calculate similarity scores between query and evidence embeddings
        scores = torch.mm(query_embedding, evidence_embeddings)
        # get indices of top k evidence embeddings with highest similarity scores
        topk_ids = torch.topk(scores, k=args.evidence_retrieval_num, dim=1).indices.tolist()

        for idx, data in enumerate(batch["datas"]):
            correct = 0 # initialize correct counter to zero
            pred_evidences = [evidence_ids[i] for i in topk_ids[idx]] # get predicted evidence IDs from top k indices
            for evidence_id in batch["evidences"][idx]:  # iterate through true evidence IDs
                if evidence_id in pred_evidences: # if true evidence ID is predicted, increment correct counter
                    correct += 1
            if correct > 0: # if at least one true evidence ID is predicted
                recall = float(correct) / len(batch["evidences"][idx])
                precision = float(correct) / len(pred_evidences)
                cur_fscore = (2 * precision * recall) / (precision + recall)
            else: # if no true evidence IDs are predicted, set f-score to 0
                cur_fscore = 0
            fscores.append(cur_fscore) # append f-score for current query to list

    print("----")
     # calculate mean f-score across all queries in validation set
    fscore = np.mean(fscores)
    # print mean f-score
    print("Evidence Retrieval F-score: %.3f" % fscore)
    
    # set encoder model to training mode
    encoder_model.train()
    # query_model.train()
    # evidence_model.train()
    
    return fscore # return mean f-score

def predict(args):
    # Load data
    dpr_setting(args)
    tok = AutoTokenizer.from_pretrained(args.model_type)
    # Create validation and evidence datasets
    test_set = ValDataset("test", tok, args.max_length)
    dev_set = ValDataset("dev", tok, args.max_length)
    evidence_set = EvidenceDataset(tok, args.max_length)
    # Create dataloaders for the validation and evidence datasets
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn)
    evidence_dataloader = DataLoader(evidence_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=evidence_set.collate_fn)
    # Load the encoder model
    encoder_model = AutoModel.from_pretrained(args.model_type)
    # query_model = AutoModel.from_pretrained(args.model_type)
    # evidence_model = AutoModel.from_pretrained(args.model_type)
    
    # Load the trained encoder model
    assert len(args.model_pt) > 0
    encoder_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))
    # query_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "query_ckpt.bin")))
    # evidence_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "evidence_ckpt.bin")))
    encoder_model.cuda()
    encoder_model.eval()

    # query_model.cuda()
    # evidence_model.cuda()
    # query_model.eval()
    # evidence_model.eval()

    # Get evidence embeddings and normalize them
    evidence_ids = []
    evidence_embeddings = []
    for batch in tqdm(evidence_dataloader):
        to_cuda(batch)
        evidence_last = encoder_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        # evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        evidence_embedding = evidence_last[:, 0, :].detach()
        evidence_embedding_cpu = nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        del evidence_embedding, evidence_last
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()
    # Perform evidence retrieval and store the results
    out_data = {}
    for batch in tqdm(test_dataloader):
        to_cuda(batch)
        query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        # query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=args.evidence_retrieval_num, dim=1).indices.tolist()
        for idx, data in enumerate(batch["datas"]):
            data["evidences"] = [evidence_ids[i] for i in topk_ids[idx]]
            out_data[batch["claim_ids"][idx]] = data
    # Write the results to file
    with open("data/retrieval-test-claims.json", 'w') as fout:
        json.dump(out_data, fout)
        
    out_data = {}
    for batch in tqdm(dev_dataloader):
        to_cuda(batch)
        query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        # query_last = query_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=args.evidence_retrieval_num, dim=1).indices.tolist()
        for idx, data in enumerate(batch["datas"]):
            data["evidences"] = [evidence_ids[i] for i in topk_ids[idx]]
            out_data[batch["claim_ids"][idx]] = data
    # Write the results to file
    with open("data/retrieval-dev-claims.json", 'w') as fout:
        json.dump(out_data, fout)

# def get_evidence_embeddings(evidence_dataloader, query_model, evidence_model):
def get_evidence_embeddings(evidence_dataloader, encoder_model):
    # set encoder_model to evaluation mode
    encoder_model.eval()
    # query_model.eval()
    # evidence_model.eval()
    
    # create empty lists to store evidence IDs and embeddings
    evidence_ids = []
    evidence_embeddings = []
    # iterate over evidence dataloader
    for batch in tqdm(evidence_dataloader):
        # move batch to CUDA device if available
        to_cuda(batch)
        # pass evidence input IDs and attention masks through encoder model to get last hidden states
        evidence_last = encoder_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        # evidence_last = evidence_model(input_ids=batch["evidence_input_ids"], attention_mask=batch["evidence_attention_mask"]).last_hidden_state
        
        # extract first token of last hidden state to get evidence embedding
        evidence_embedding = evidence_last[:, 0, :].detach()
        # normalize evidence embeddings
        evidence_embedding_cpu = nn.functional.normalize(evidence_embedding, p=2, dim=1).cpu()
        # free up GPU memory by deleting unnecessary variables
        del evidence_embedding, evidence_last
        # append evidence embeddings and IDs to lists
        evidence_embeddings.append(evidence_embedding_cpu)
        evidence_ids.extend(batch["evidences_ids"])
    # concatenate evidence embeddings and transpose the tensor to get (embedding_dim x num_evidences) shape
    evidence_embeddings = torch.cat(evidence_embeddings, dim=0).t()
    # return evidence embeddings and IDs
    return evidence_embeddings, evidence_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--model_pt", default="", type=str, help="built model path")
    parser.add_argument("-p", "--predict", action="store_true", help="prediction")
    args = parser.parse_args()
    #perform prediction
    if args.predict:
        predict(args)
    #train model
    else:
        train(args)

