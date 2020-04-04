import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np 


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["intents"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        targets_argmax = np.argmax(targets.cpu().detach().numpy(),axis=1).tolist()
        outputs_argmax = np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),axis=1).tolist()
        # print(targets_argmax)
        # print(outputs_argmax)
        # print(loss.cpu().detach().numpy())


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["intents"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, targets)
            targets_argmax = np.argmax(targets.cpu().detach().numpy(),axis=1).tolist()
            outputs_argmax = np.argmax(torch.sigmoid(outputs).cpu().detach().numpy(),axis=1).tolist()
            fin_targets.extend(targets_argmax)
            fin_outputs.extend(outputs_argmax)
    return fin_outputs, fin_targets