# SCINLI - Inferência de Linguagem Natural Científica com RoBERTa

Este projeto implementa um modelo de **Inferência de Linguagem Natural (NLI)** voltado para o domínio **científico**, utilizando o dataset [SCINLI](https://huggingface.co/datasets/tasksource/scinli) e o modelo pré-treinado **RoBERTa-base** da biblioteca Transformers.  
Ele é inspirado no artigo **"Co-Training for Low Resource Scientific Natural Language Inference" (ACL 2024)**, que propõe o uso de *Weighted Co-Training (WCT)* para melhorar o aprendizado em cenários com poucos dados rotulados.

---

##  Objetivo do Projeto

Treinar um modelo capaz de **entender a relação entre duas sentenças científicas**, determinando se:
- Uma **implica** a outra (*entailment*);
- Elas se **contradizem** (*contrasting*);
- São **neutras** (sem relação direta);
- Ou envolvem **raciocínio científico** (*reasoning*).

---

## Instalação e Execução

### Clonar o repositório
```bash
git clone https://github.com/SEU_USUARIO/scinli-wct.git
cd scinli-wct

```

## Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/macOS
```

## Instalar dependências
```bash
pip install -r requirements.txt
```

## Executar o treinamento

```bash
cd scinli-wct
python train_wct.py
```


# O modelo será treinado e você verá saídas como:
```bash
Carregando dataset SCINLI...
Selecionando pequeno subconjunto rotulado (Dl)...
Conjunto Dl criado com 400 amostras.
Tokenizando textos...
Treinando modelo base...
Epoch 1: 100%|██████████████████████████████████████████████| 50/50 [01:18<00:00, 1.56s/it]
Loss médio: 1.3981
✅ Treinamento inicial concluído!
```

🧩 Funcionamento do Código
Entrada

O modelo recebe duas sentenças científicas:

    sentence1: a premissa

    sentence2: a hipótese

Exemplo:

sentence1 = "Increasing temperature raises reaction rate."
sentence2 = "Reactions are faster at higher temperatures."

## Processamento

O modelo RoBERTa-base analisa as duas sentenças simultaneamente e gera representações vetoriais de contexto, entendendo o significado das palavras e as relações entre as sentenças.
## Classificação

A camada final (classifier) transforma a representação em 4 probabilidades, uma para cada relação:
Classe	Significado	Exemplo
0	Contrasting	“O tratamento reduziu a dor.” / “A dor aumentou.”
1	Reasoning	“A pressão aumentou.” / “A temperatura pode ter subido.”
2	Entailment	“O sol aquece a Terra.” / “A Terra recebe calor do sol.”
3	Neutral	“A amostra foi aquecida.” / “A mistura foi resfriada.”

A classe com maior probabilidade é a previsão final do modelo.
## Aprendizado

Durante o treino:

O modelo faz previsões;

Calcula o erro (loss) comparando com o rótulo real;

O otimizador AdamW ajusta os pesos internos;

O processo se repete por várias épocas, reduzindo o loss e melhorando a precisão.

Arquivo config.yaml

```bash

model_name: "roberta-base"
batch_size: 8
lr: 2e-5
epochs_init: 1
epochs_cotraining: 1
epochs_finetune: 1
max_length: 128
`
Como ajustar para um treinamento mais completo

O arquivo config.yaml controla todo o comportamento do treinamento.  
Se o usuário quiser um treinamento mais longo, mais preciso ou com mais dados, basta modificar alguns parâmetros ali:
```
---

### Parâmetros principais

| Parâmetro | Função | Valor padrão | Como ajustar para mais treino |
|------------|---------|---------------|-------------------------------|
| `epochs_init` | Quantas vezes o modelo vê o subset rotulado `Dl` | `1` | Aumente para `3`–`10` para treinar mais |
| `per_class_small_Dl` | Quantas amostras por classe usar no subset rotulado | `100` | Aumente para `300`–`1000` para mais dados |
| `batch_size` | Quantas amostras por iteração | `8` | Pode subir para `16` ou `32` (se tiver GPU) |
| `lr` | Taxa de aprendizado | `2e-5` | Pode reduzir (ex: `1e-5`) se aumentar as épocas |
| `epochs_finetune` | Épocas adicionais de fine-tuning com todo o dataset | `1` | Aumente para `3`–`5` para melhorar desempenho |
| `device` | CPU ou GPU | `"cuda"` | Troque para `"cpu"` se não tiver GPU NVIDIA |

---

### Recomendações práticas

- **Quer mais qualidade:** aumente `epochs_init` e `per_class_small_Dl`.  
- **Quer treinar mais rápido:** reduza `per_class_small_Dl` e use `batch_size` pequeno.  
- **Quer o máximo possível:** use `epochs_init: 5`, `epochs_finetune: 3`, e `per_class_small_Dl: 500`.  
- **Sem GPU:** mude `device: "cpu"` — o código funciona igual, apenas mais lento.

---

###  Exemplo de configuração “forte”

```yaml
model_name: "roberta-base"
batch_size: 16
lr: 1e-5
epochs_init: 5
epochs_cotraining: 1
epochs_finetune: 3
max_length: 128
seed: 42
device: "cuda"
per_class_small_Dl: 500
seed: 42
device: "cuda"     # ou "cpu"
per_class_small_Dl: 100
```
