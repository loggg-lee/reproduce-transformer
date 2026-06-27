import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

raw_dataset = load_dataset('parquet', data_files={'train': "/home/binle/NEW-SEDD/Score-Entropy-Discrete-Diffusion-main/dataset_en-zh/zh-en/**/*.parquet"}, split='train')
raw_dataset = raw_dataset.shuffle(seed=42).select(range(500000))
print(f"Loaded dataset with {len(raw_dataset)} samples.")

#tokenizer_path = "/home/binle/NEW-SEDD/Score-Entropy-Discrete-Diffusion-main/NLP_TOKEN/tokenizer_config.json"
tokenizer = AutoTokenizer.from_pretrained("/home/binle/NEW-SEDD/Score-Entropy-Discrete-Diffusion-main/NLP_TOKEN")
print(raw_dataset.features)
# 编码函数
def encode(example):
    translation_pair = example['translation']
    src = tokenizer(translation_pair['en'], padding='max_length', truncation=True, max_length=100)
    tgt = tokenizer(translation_pair['zh'], padding='max_length', truncation=True, max_length=100)
    return {'src_input_ids': src['input_ids'], 'tgt_input_ids': tgt['input_ids']}

tokenized_dataset = raw_dataset.map(encode, remove_columns=raw_dataset.column_names)

# PyTorch Dataset 包装
class WMTDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = torch.tensor(self.data[idx]['src_input_ids'], dtype=torch.long)
        tgt = torch.tensor(self.data[idx]['tgt_input_ids'], dtype=torch.long)
        return src, tgt

train_data = WMTDataset(tokenized_dataset)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):#多头注意力机制
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):#计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):#将张量拆分为多头/合并多头以支持并行计算。
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):#前向传播
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))#拆分多头

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)#计算注意力
        output = self.W_o(self.combine_heads(attn_output))#输出
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):#模型整合
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length,device=tgt.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))


        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


sc_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 256
dropout = 0.1
src_vocab_size = len(tokenizer)
tgt_vocab_size = len(tokenizer)
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)


transformer = transformer.to(device)

# Generate random sample data
#src_data = torch.randint(1, src_vocab_size, (64, max_seq_length), device=device)
#tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length), device=device)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = transformer(src_data, tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
for epoch in range(100):
    total_loss = 0.0
    for batch in train_loader:  # 迭代DataLoader获取真实批次数据
        src_data, tgt_data = batch  # 从批次中获取源数据和目标数据
        src_data = src_data.to(device)  # 转移到设备
        tgt_data = tgt_data.to(device)

        optimizer.zero_grad()
        # 目标输入取前n-1个token，目标输出取后n-1个token（用于teacher forcing）
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # 训练结束后保存模型（可选）
    # torch.save(transformer.state_dict(), "transformer_translation.pt")


    # 定义贪婪解码推理函数
    def greedy_decode(model, src_sentence, tokenizer, max_len=100, device='device'
                                                                        ):
        model.eval()
        with torch.no_grad():
            src_tokens = tokenizer(src_sentence, return_tensors="pt", padding='max_length',
                                   truncation=True, max_length=max_len)
            src_input = src_tokens['input_ids'].to(device)

            # 开始符：尝试用BOS token，如果没有则使用CLS token
            if tokenizer.bos_token_id is not None:
                start_token_id = tokenizer.bos_token_id
            elif tokenizer.cls_token_id is not None:
                start_token_id = tokenizer.cls_token_id
            else:
                start_token_id = tokenizer.convert_tokens_to_ids('<s>')  # 可换成自定义起始符

            if tokenizer.eos_token_id is not None:
                end_token_id = tokenizer.eos_token_id
            elif tokenizer.sep_token_id is not None:
                end_token_id = tokenizer.sep_token_id
            else:
                end_token_id = tokenizer.convert_tokens_to_ids('</s>')  # 可换成自定义结束符

            tgt_input = torch.tensor([[start_token_id]], device=device)

            for _ in range(max_len):
                output = model(src_input, tgt_input)
                next_token_logits = output[:, -1, :]  # 取最后一位预测
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # 贪婪解码

                if next_token.item() == end_token_id:
                    break
                tgt_input = torch.cat([tgt_input, next_token], dim=-1)

            translated = tokenizer.decode(tgt_input.squeeze(), skip_special_tokens=True)
            return translated


    # 示例：执行一次翻译推理
    test_sentence = "Panama wishes here and now to reaffirm its commitment to the principle of self-determination of peoples, articulated in the Charter of the United Nations, and to the use of mechanisms for the peaceful settlement of conflicts."
    translated = greedy_decode(transformer, test_sentence, tokenizer, max_len=256, device=device)
    print(f"\n[Translation Test]")
    print(f"Source: {test_sentence}")
    print(f"Translated: {translated}")

    # --- 新增：将翻译结果写入文件 ---
    try:
        output_file_path = "/home/binle/NEW-SEDD/Score-Entropy-Discrete-Diffusion-main/transformer_trans.txt"
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(f"--- Epoch {epoch + 1} ---\n")
            f.write(f"Source: {test_sentence}\n")
            f.write(f"Translated: {translated}\n")
            f.write("\n")
    except Exception as e:
        print(f"警告：无法写入翻译结果到文件 {output_file_path}。错误: {e}")
    # ------------------------------------

