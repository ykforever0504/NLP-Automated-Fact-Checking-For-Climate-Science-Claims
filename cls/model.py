from torch import nn
from transformers import AutoModel


class CLSModel(nn.Module):
    def __init__(self, pre_encoder):

        super(CLSModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        #last layer classification
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, input_ids, attention_mask):
        texts_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # first token
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)
        return logits
