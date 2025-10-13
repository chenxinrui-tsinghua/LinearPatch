from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
import os

def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get wikitext2")
    traindata = load_dataset("wikitext", 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset("wikitext", 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_ptb(tokenizer, nsamples, val_size, seed, seqlen, test_only):
    print("get_ptb")
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    if test_only:
        return testenc
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc



def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")

    traindata = load_dataset(
        'allenai/c4', data_files={'train': '/datasets/c4/c4-train.00000-of-01024.json'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': '/datasets/c4/c4-validation.00000-of-00008.json'},
        split='validation'
    )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc 

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader 

def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")

    loacal_dataset = "/datasets/RedPajama-Data-1T-Sample"
    traindata = load_dataset(loacal_dataset,split='train')

    random.seed(seed)
    traindata = traindata.shuffle(seed=seed) 
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'c4' in name:
        return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'ptb' in name:
        return get_ptb(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        nlls = []
        if hasattr(model,'lm_head'): # and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
    return results


class InferenceDataset(Dataset):
    def __init__(self, save_dir, cache_bs, size):
        self.save_dir = save_dir
        self.len = size
        self.cache_bs = cache_bs
        self.data_idx = 0
        self.cache = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        data_idx = idx % self.cache_bs
        if data_idx == 0:
            file_idx = idx // self.cache_bs
            cache_path = os.path.join(self.save_dir, f'data_cache_{file_idx}.pt')
            self.cache = torch.load(cache_path)
            self.data_idx += self.cache_bs

        return self.cache[data_idx]

def get_offline_dataset(model, trainloader, source_layer, save_dir, args, dev, cache_bs=500):

    if os.path.exists(save_dir):
        inference_dataset = InferenceDataset(save_dir, cache_bs, len(trainloader))
        target_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)
        return target_loader

    os.makedirs(save_dir, exist_ok=True)
    model.to(dev).half().eval()

    with torch.no_grad():
        batch_list = []
        for i, batch in tqdm(enumerate(trainloader), total=len(trainloader), desc="Generating offline dataset"):
            t_outputs = model(batch[0].to(dev))[0]
            topk_logits, topk_indices = torch.topk(t_outputs, 50, dim=-1)
            t_probs = torch.softmax(topk_logits, dim=-1)  # [1, 2048, 100]
            data = (batch[0].cpu(), t_probs.half().cpu(), topk_indices[0].cpu())
            batch_list.append(data)

            if (i + 1) % cache_bs == 0 or i == len(trainloader) - 1:
                file_idx = i // cache_bs
                torch.save(batch_list, os.path.join(save_dir, f'data_cache_{file_idx}.pt'))
                batch_list = []

    inference_dataset = InferenceDataset(save_dir, cache_bs, len(trainloader))
    target_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=True)

    return target_loader