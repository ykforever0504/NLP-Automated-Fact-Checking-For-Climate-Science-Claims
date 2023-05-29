import torch
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import to_cuda, ValDataset, EvidenceDataset
from config import dpr_setting


def generate_negative_samples(args):
    # Load data
    dpr_setting(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    val_dataset = ValDataset("train", tokenizer, args.max_length)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn)

    # Build encoder model
    encoder_model = AutoModel.from_pretrained(args.model_type)

    if args.model_pt:
        encoder_model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt, "best_ckpt.bin")))
    encoder_model.to('cuda')
    encoder_model.eval()

    # Load evidence embeddings
    evidence_embeddings = torch.load("data_t/evidence_embeddings")
    evidence_ids = torch.load("data_t/evidence_ids")

    # Generate negative samples for each claim
    out_data = {}
    for batch in tqdm(dataloader):
        to_cuda(batch)
        query_last = encoder_model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"]).last_hidden_state
        query_embedding = query_last[:, 0, :]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu()
        scores = torch.mm(query_embedding, evidence_embeddings)
        topk_ids = torch.topk(scores, k=64, dim=1).indices.tolist()

        for idx, data in enumerate(batch["datas"]):
            negative_evidences = []
            for i in topk_ids[idx]:
                if evidence_ids[i] not in batch["evidences"][idx]:
                    negative_evidences.append(evidence_ids[i])
            data["negative_evidences"] = negative_evidences
            out_data[batch["claim_ids"][idx]] = data

    # Save the output to file
    with open("data/train-claims-negative.json", 'w') as fout:
        json.dump(out_data, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate negative samples for training set')
    parser.add_argument("--model_pt", default="", type=str, help="path to the pre-trained encoder model checkpoint")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str, help="pre-trained model type")
    parser.add_argument("--max_length", default=256, type=int, help="maximum sequence length")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for processing the data")
    args = parser.parse_args()

    generate_negative_samples(args)


