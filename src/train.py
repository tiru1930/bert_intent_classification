import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from bidict import bidict

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def encode_intents(df):
    intents_bidict = bidict({i:intent.strip() for i, intent in enumerate(df.category.unique())})
    return intents_bidict 

def run():
    df_train = pd.read_csv(config.TRAINING_FILE).fillna("none")
    intents_idx = encode_intents(df_train)
    df_train.category = df_train.category.apply(
        lambda x: intents_idx.inverse[x.strip()]
    )
    df_valid = pd.read_csv(config.TESTING_FILE).fillna("none")    
    df_valid.category = df_valid.category.apply(
        lambda x: intents_idx.inverse[x.strip()]
    )

    np.save("intents_idx.npy",intents_idx)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        text=df_train.text.values,
        intent=df_train.category.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values,
        intent=df_valid.category.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BERTBaseUncased(len(intents_idx.keys()))
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()