import math
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD, UNK, CLS, SEP, MASK = SPECIAL_TOKENS


def simple_tokenize(text: str) -> List[str]:
    pattern = r"\[(?:pad|unk|cls|sep|mask)\]|\w+|[^\w\s]"
    tokens = re.findall(pattern, text.lower(), flags=re.UNICODE)
    mapping = {"[pad]": PAD, "[unk]": UNK, "[cls]": CLS, "[sep]": SEP, "[mask]": MASK}
    return [mapping.get(tok, tok) for tok in tokens]


class SimpleTokenizer:
    def __init__(self, documents: List[List[str]]):
        vocab = set()
        for doc in documents:
            for sent in doc:
                vocab.update(simple_tokenize(sent))
        self.itos = SPECIAL_TOKENS + sorted(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(tok, self.stoi[UNK]) for tok in simple_tokenize(text)]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


@dataclass
class PairExample:
    sentence_a: str
    sentence_b: str
    is_next: int


class BertPretrainingDataset:
    def __init__(self, documents: List[List[str]], tokenizer: SimpleTokenizer, max_len: int = 24):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.all_sentences = [sent for doc in documents for sent in doc]
        self.pairs = self._build_pairs()

    def _build_pairs(self) -> List[PairExample]:
        pairs: List[PairExample] = []
        for doc_idx, doc in enumerate(self.documents):
            other_sentences = [sent for j, other_doc in enumerate(self.documents) if j != doc_idx for sent in other_doc]
            for i in range(len(doc) - 1):
                a = doc[i]
                b_true = doc[i + 1]
                pairs.append(PairExample(a, b_true, 1))
                b_false = random.choice(other_sentences)
                pairs.append(PairExample(a, b_false, 0))
        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _truncate_pair(self, a_ids: List[int], b_ids: List[int]) -> Tuple[List[int], List[int]]:
        max_pair_len = self.max_len - 3
        while len(a_ids) + len(b_ids) > max_pair_len:
            if len(a_ids) > len(b_ids):
                a_ids.pop()
            else:
                b_ids.pop()
        return a_ids, b_ids

    def _apply_mlm(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        mlm_input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)

        candidate_positions = [
            i for i, tok_id in enumerate(input_ids)
            if tok_id not in {
                self.tokenizer.stoi[CLS],
                self.tokenizer.stoi[SEP],
                self.tokenizer.stoi[PAD],
            }
        ]

        num_to_mask = max(1, int(round(0.15 * len(candidate_positions))))
        random.shuffle(candidate_positions)
        masked_positions = candidate_positions[:num_to_mask]

        vocab_ids = list(range(self.tokenizer.vocab_size))
        special_ids = {self.tokenizer.stoi[t] for t in SPECIAL_TOKENS}
        normal_vocab_ids = [i for i in vocab_ids if i not in special_ids]

        for pos in masked_positions:
            original_id = input_ids[pos]
            labels[pos] = original_id
            p = random.random()
            if p < 0.8:
                mlm_input_ids[pos] = self.tokenizer.stoi[MASK]
            elif p < 0.9:
                mlm_input_ids[pos] = random.choice(normal_vocab_ids)
            else:
                mlm_input_ids[pos] = original_id

        return mlm_input_ids, labels

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        a_ids = self.tokenizer.encode(pair.sentence_a)
        b_ids = self.tokenizer.encode(pair.sentence_b)
        a_ids, b_ids = self._truncate_pair(a_ids, b_ids)

        input_ids = [self.tokenizer.stoi[CLS]] + a_ids + [self.tokenizer.stoi[SEP]] + b_ids + [self.tokenizer.stoi[SEP]]
        token_type_ids = [0] * (len(a_ids) + 2) + [1] * (len(b_ids) + 1)

        input_ids, mlm_labels = self._apply_mlm(input_ids)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.stoi[PAD]] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            mlm_labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "nsp_labels": torch.tensor(pair.is_next, dtype=torch.long),
        }


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int, type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(x))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size precisa ser divisível por num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(padding_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class BertLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, attention_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MiniBertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 128,
        max_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        cls = torch.tanh(self.pooler(x[:, 0]))
        return x, cls


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size: int, embedding_weight: nn.Parameter):
        super().__init__()
        vocab_size = embedding_weight.shape[0]
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.embedding_weight = embedding_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return torch.matmul(x, self.embedding_weight.t()) + self.decoder_bias


class BertPretrainingModel(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 24):
        super().__init__()
        self.bert = MiniBertEncoder(vocab_size=vocab_size, max_len=max_len)
        self.mlm_head = BertOnlyMLMHead(
            hidden_size=64,
            embedding_weight=self.bert.embeddings.word_embeddings.weight,
        )
        self.nsp_head = nn.Linear(64, 2)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits


TRAINING_DOCS = [
    [
        "Transformers usam atenção para modelar dependências longas.",
        "O BERT é um encoder bidirecional treinado com masked language modeling.",
        "O token CLS resume a sequência para tarefas de classificação.",
    ],
    [
        "A USP desenvolve pesquisa avançada em inteligência artificial.",
        "Modelos de linguagem precisam de grandes volumes de texto para pré treino.",
        "Fine tuning adapta o modelo para tarefas específicas.",
    ],
    [
        "Redes neurais aprendem representações distribuídas dos dados.",
        "O mecanismo de self attention compara consultas chaves e valores.",
        "Embeddings de posição indicam a ordem dos tokens na frase.",
    ],
    [
        "O next sentence prediction usa pares de sentenças relacionadas ou não.",
        "O objetivo masked language modeling tenta reconstruir tokens ocultos.",
        "A combinação dos dois objetivos produz um pré treino didático do BERT.",
    ],
]


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


@torch.no_grad()
def predict_mask(model: BertPretrainingModel, tokenizer: SimpleTokenizer, sentence_a: str, sentence_b: str = ""):
    a_ids = tokenizer.encode(sentence_a)
    b_ids = tokenizer.encode(sentence_b) if sentence_b else []

    input_ids = [tokenizer.stoi[CLS]] + a_ids + [tokenizer.stoi[SEP]]
    token_type_ids = [0] * len(input_ids)
    if b_ids:
        input_ids += b_ids + [tokenizer.stoi[SEP]]
        token_type_ids += [1] * (len(b_ids) + 1)

    if len(input_ids) > 24:
        input_ids = input_ids[:24]
        token_type_ids = token_type_ids[:24]

    attention_mask = [1] * len(input_ids)
    while len(input_ids) < 24:
        input_ids.append(tokenizer.stoi[PAD])
        token_type_ids.append(0)
        attention_mask.append(0)

    input_ids_tensor = torch.tensor([input_ids], device=DEVICE)
    token_type_tensor = torch.tensor([token_type_ids], device=DEVICE)
    attention_tensor = torch.tensor([attention_mask], device=DEVICE)

    mlm_logits, _ = model(input_ids_tensor, token_type_tensor, attention_tensor)
    tokens = tokenizer.decode_ids(input_ids)
    try:
        mask_pos = tokens.index(MASK)
    except ValueError:
        return []

    probs = torch.softmax(mlm_logits[0, mask_pos], dim=-1)
    values, indices = torch.topk(probs, k=5)
    return [(tokenizer.itos[i.item()], values[j].item()) for j, i in enumerate(indices)]


@torch.no_grad()
def predict_nsp(model: BertPretrainingModel, tokenizer: SimpleTokenizer, sentence_a: str, sentence_b: str):
    a_ids = tokenizer.encode(sentence_a)
    b_ids = tokenizer.encode(sentence_b)

    while len(a_ids) + len(b_ids) > 21:
        if len(a_ids) > len(b_ids):
            a_ids.pop()
        else:
            b_ids.pop()

    input_ids = [tokenizer.stoi[CLS]] + a_ids + [tokenizer.stoi[SEP]] + b_ids + [tokenizer.stoi[SEP]]
    token_type_ids = [0] * (len(a_ids) + 2) + [1] * (len(b_ids) + 1)
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < 24:
        input_ids.append(tokenizer.stoi[PAD])
        token_type_ids.append(0)
        attention_mask.append(0)

    input_ids_tensor = torch.tensor([input_ids], device=DEVICE)
    token_type_tensor = torch.tensor([token_type_ids], device=DEVICE)
    attention_tensor = torch.tensor([attention_mask], device=DEVICE)

    _, nsp_logits = model(input_ids_tensor, token_type_tensor, attention_tensor)
    probs = torch.softmax(nsp_logits[0], dim=-1)
    return {
        "nao_e_proxima": probs[0].item(),
        "e_proxima": probs[1].item(),
    }


def train_didactic_bert(epochs: int = 40, batch_size: int = 8):
    tokenizer = SimpleTokenizer(TRAINING_DOCS)
    dataset = BertPretrainingDataset(TRAINING_DOCS, tokenizer, max_len=24)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = BertPretrainingModel(vocab_size=tokenizer.vocab_size, max_len=24).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mlm = 0.0
        total_nsp = 0.0

        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            mlm_logits, nsp_logits = model(
                batch["input_ids"],
                batch["token_type_ids"],
                batch["attention_mask"],
            )

            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, tokenizer.vocab_size),
                batch["mlm_labels"].view(-1),
                ignore_index=-100,
            )
            nsp_loss = F.cross_entropy(nsp_logits, batch["nsp_labels"])
            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mlm += mlm_loss.item()
            total_nsp += nsp_loss.item()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            n_batches = len(loader)
            print(
                f"epoch={epoch:02d} "
                f"loss={total_loss / n_batches:.4f} "
                f"mlm={total_mlm / n_batches:.4f} "
                f"nsp={total_nsp / n_batches:.4f}"
            )

    return model, tokenizer


def main():
    print(f"Executando em: {DEVICE}")
    model, tokenizer = train_didactic_bert()

    model.eval()

    print("\n=== Demonstração MLM ===")
    demo_sentence = "o bert é um [MASK] bidirecional treinado com masked language modeling ."
    top5 = predict_mask(model, tokenizer, demo_sentence)
    for token, prob in top5:
        print(f"token={token:<15} prob={prob:.4f}")

    print("\n=== Demonstração NSP ===")
    s1 = "a usp desenvolve pesquisa avançada em inteligência artificial ."
    s2_true = "modelos de linguagem precisam de grandes volumes de texto para pré treino ."
    s2_false = "embeddings de posição indicam a ordem dos tokens na frase ."

    print("Par verdadeiro:", predict_nsp(model, tokenizer, s1, s2_true))
    print("Par falso:    ", predict_nsp(model, tokenizer, s1, s2_false))

    print("\nObservações:")
    print("1. Esta é uma implementação didática, pequena e treinada em corpus minúsculo.")
    print("2. Ela preserva as ideias centrais do BERT: encoder bidirecional, MLM e NSP.")
    print("3. Para simplificar, usei tokenização por palavras e pontuação, não WordPiece.")


if __name__ == "__main__":
    main()
