import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np, random, yaml
from utils import set_seed, NLIDataset, LABEL2ID
from tqdm import tqdm

def prepare_dataset(tokenizer, split, max_length):
    sentences1 = [x['sentence1'] for x in split]
    sentences2 = [x['sentence2'] for x in split]
    enc = tokenizer(sentences1, sentences2, truncation=True, padding='max_length', max_length=max_length)
    labels = [LABEL2ID[x['label']] for x in split]
    return enc, labels

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")

    print("Carregando dataset SCINLI")
    ds = load_dataset("tasksource/scinli")
    train, val, test = ds["train"], ds["validation"], ds["test"]

    print("Selecionando pequeno subconjunto rotulado (Dl).")
    per_class = cfg['per_class_small_Dl']
    selected, counts = [], {lab: 0 for lab in LABEL2ID.keys()}
    for i, ex in enumerate(train):
        lab = ex['label']
        if counts[lab] < per_class:
            selected.append(i)
            counts[lab] += 1
        if all(c >= per_class for c in counts.values()):
            break
    Dl = train.select(selected)
    print(f"Conjunto Dl criado com {len(Dl)} amostras.")

    print("Tokenizando textos")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    enc_Dl, labels_Dl = prepare_dataset(tokenizer, Dl, cfg['max_length'])
    ds_Dl = NLIDataset(enc_Dl, labels_Dl)
    dl_Dl = DataLoader(ds_Dl, batch_size=cfg['batch_size'], shuffle=True)

    print("Treinando modelo base")
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name'], num_labels=4).to(device)
    optim = AdamW(model.parameters(), lr=float(cfg['lr']))


    for epoch in range(cfg['epochs_init']):
        model.train()
        total_loss = 0
        for batch in tqdm(dl_Dl, desc=f"Epoch {epoch+1}"):
            inputs = {k:v.to(device) for k,v in batch.items()}
            out = model(**inputs)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += out.loss.item()
        print(f"Loss médio: {total_loss / len(dl_Dl):.4f}")

    print("Treinamento inicial concluido")


    print("FFine-tuning com todo o dataset")
    enc_train, labels_train = prepare_dataset(tokenizer, train, cfg['max_length'])
    ds_train = NLIDataset(enc_train, labels_train)
    dl_train = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=True)
    for epoch in range(cfg['epochs_finetune']):
        model.train()
        total_loss = 0
        for batch in tqdm(dl_train, desc=f"Fine-tuning Epoch {epoch+1}"):
            inputs = {k:v.to(device) for k,v in batch.items()}
            out = model(**inputs)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += out.loss.item()
        print(f"Fine-tuning Loss médio: {total_loss / len(dl_train):.4f}")
        print("SSalvando modelo treinado.")
    model.save_pretrained("outputs/roberta_scinli")
    tokenizer.save_pretrained("outputs/roberta_scinli")
    print("Modelo salvo com sucesso!")


if __name__ == "__main__":
    main()
