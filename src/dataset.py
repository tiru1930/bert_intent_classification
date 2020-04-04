import config
import torch
from sklearn.preprocessing import OneHotEncoder


class BERTDataset:
    def __init__(self, text, intent):
        intent_ohe = OneHotEncoder()
        self.text = text
        self.intent = intent_ohe.fit_transform(intent.reshape(-1,1)).toarray()
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'intents': torch.tensor(self.intent[item], dtype=torch.long)
        }