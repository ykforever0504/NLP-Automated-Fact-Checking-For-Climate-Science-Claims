from torch.utils.data import Dataset
import json
import random
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english'))

def process(text):
    # Convert to lowercase
    text = text.lower()

    # # Remove stop words
    # tokens = text.split()
    # tokens = [token for token in tokens if token not in stop_words]
    # text = " ".join(tokens)

    return text


def to_cuda(batch):
    for n in batch.keys():
        if n in ["query_input_ids", "evidence_input_ids", "query_attention_mask", "evidence_attention_mask"]:
            batch[n] = batch[n].cuda()


class TrainDataset(Dataset):
    def __init__(self, mode, tok, evidence_samples, max_length, negative=True):
        # initialize variables and load dataset
        if negative:
            f = open("data/train-claims-negative.json", "r")
        else:
            f = open("data/%s-claims.json"%(mode), "r")
        self.dataset = json.load(f)
        f.close()
        self.using_negative = negative
        f = open("data/evidence.json", "r")
        self.evidences = json.load(f)
        f.close()

        self.mode = mode
        self.tokenizer = tok
        self.max_length = max_length
        self.evidence_samples = evidence_samples
        self.claim_ids = list(self.dataset.keys())
        self.evidence_ids = list(self.evidences.keys())

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        # get processed query and evidences
        data = self.dataset[self.claim_ids[idx]]
        processed_query = process(data["claim_text"])
        evidences = []
        for evidence_id in data["evidences"]:
            evidences.append(evidence_id)
        # get processed query and evidences
        if self.using_negative:
            negative_evidences = data["negative_evidences"]
            return [processed_query, evidences, negative_evidences]
        else:
            return [processed_query, evidences]
    # initialize variables for queries, evidences, and labels
    def collate_fn(self, batch):
        queries = []
        evidences = []
        labels = []
        # initialize variables for queries, evidences, and labels
        if self.using_negative:
            negative_evidences = []
            for query, evidence, negative_evidence in batch:
                queries.append(query)
                evidences.extend(evidence)
                negative_evidences.extend(negative_evidence)
                labels.append(len(evidence))
            evidences.extend(negative_evidences)
        # otherwise just get evidences
        else:
            for query, evidence in batch:
                queries.append(query)
                evidences.extend(evidence)
                labels.append(len(evidence))
        # truncate evidences if there are too many
        cnt = len(evidences)
        if cnt > self.evidence_samples:
            evidences = evidences[:self.evidence_samples]
        # get evidence text
        evidences_text = [process(self.evidences[evidence_id]) for evidence_id in evidences]
       # add more random evidences if there aren't enough
        while cnt < self.evidence_samples:
            evidence_id = random.choice(self.evidence_ids)
            while evidence_id in evidences:
                evidence_id = random.choice(self.evidence_ids)
            evidences.append(evidence_id)
            evidences_text.append(process(self.evidences[evidence_id]))
            cnt += 1
        # tokenize queries and evidences
        query_text = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        evidence_text = self.tokenizer(
            evidences_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        # return batch encoding
        return {
            "query_input_ids": query_text.input_ids,
            "evidence_input_ids" : evidence_text.input_ids,
            "query_attention_mask": query_text.attention_mask,
            "evidence_attention_mask" : evidence_text.attention_mask,
            "labels" : labels
        }


class EvidenceDataset(Dataset):
    def __init__(self, tok, max_length=512):
        """
        A PyTorch Dataset that loads the evidence dataset.

        :param tok: A tokenizer object from the transformers library.
        :param max_length: The maximum sequence length.
        """
        self.max_length = max_length

        # Load the evidence data from file
        with open("data/evidence.json", "r") as f:
            self.evidences = json.load(f)

        self.tokenizer = tok
        self.evidences_ids = list(self.evidences.keys())

    def __len__(self):
        """
        Return the number of examples in the dataset.
        """
        return len(self.evidences_ids)

    def __getitem__(self, idx):
        """
        Return an example from the dataset.

        :param idx: The index of the example to retrieve.
        :return: A list containing the evidence ID and evidence text.
        """
        evidences_id = self.evidences_ids[idx]
        evidence = self.evidences[evidences_id]
        return [evidences_id, evidence]

    def collate_fn(self, batch):
        """
        Process a batch of examples.

        :param batch: A list of examples.
        :return: A dictionary containing the processed batch.
        """
        evidences_ids = []
        evidences = []

        # Process each example in the batch
        for evidences_id, evidence in batch:
            evidences_ids.append(evidences_id)
            evidences.append(process(evidence))

        evidences_text = self.tokenizer(
            evidences,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        # Return the processed batch as a dictionary
        return {
            "evidence_input_ids": evidences_text.input_ids,
            "evidence_attention_mask": evidences_text.attention_mask,
            "evidences_ids": evidences_ids,
        }


class ValDataset(Dataset):
    def __init__(self, mode, tokenizer, max_length=512):
        self.max_length = max_length
        if mode != "test":
            with open(f"data/{mode}-claims.json", "r") as f:
                self.dataset = json.load(f)
        else:
            with open("data/test-claims-unlabelled.json", "r") as f:
                self.dataset = json.load(f)

        self.tokenizer = tokenizer
        self.claim_ids = list(self.dataset.keys())
        self.mode = mode

    def __len__(self):
        return len(self.claim_ids)

    def __getitem__(self, idx):
        claim_id = self.claim_ids[idx]
        data = self.dataset[claim_id]
        claim_text = process(data["claim_text"])  # preprocess claim text
        return [claim_text, data, claim_id]

    def collate_fn(self, batch):
        queries = []
        datas = []
        evidences = []
        claim_ids = []
        for query, data, claim_id in batch:
            queries.append(query)
            datas.append(data)
            if self.mode != "test":
                evidences.append(data["evidences"])
            claim_ids.append(claim_id)

        query_encoding = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["query_input_ids"] = query_encoding["input_ids"]
        batch_encoding["query_attention_mask"] = query_encoding["attention_mask"]

        batch_encoding["datas"] = datas
        batch_encoding["claim_ids"] = claim_ids
        if self.mode != "test":
            batch_encoding["evidences"] = evidences
        return batch_encoding
