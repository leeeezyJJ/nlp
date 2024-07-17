# 【Datawhale AI 夏令营】基于术语词典干预的机器翻译挑战赛_02
#Datawhale #nlp #机器翻译

## Task2--从 baseline 代码详解入门深度学习
### 基于Seq2Seq的Baseline详解
1. 配置环境：
    - torchtext ：是一个用于自然语言处理（NLP）任务的库，它提供了丰富的功能，包括数据预处理、词汇构建、序列化和批处理等，特别适合于文本分类、情感分析、机器翻译等任务
    - jieba：是一个中文分词库，用于将中文文本切分成有意义的词语
    - sacrebleu：用于评估机器翻译质量的工具，主要通过计算BLEU（Bilingual Evaluation Understudy）得分来衡量生成文本与参考译文之间的相似度
    - spacy：是一个强大的自然语言处理库，支持70+语言的分词与训练
  其中 torchtext、jiaba、sacrebleu 三个库可直接通过 pip 安装：
      
          !pip install torchtext    
          !pip install jieba
          !pip install sacrebleu

  spacy 库要针对不同环境进行选择，这里我们选择 spacy 用于英文的 tokenizer (分词，就是将句子、段落、文章这种长文本，分解为以字词为单位的数据结构，方便后续的处理分析工作)
  ![image](https://github.com/user-attachments/assets/bc2a4674-ed3c-46d8-be9e-a950f35def1f)

          !pip install -U pip setuptools wheel
          !pip install -U 'spacy[cuda11x]'
          !python -m spacy download en_core_web_trf

  需要注意的是，使用命令!python -m spacy download en_core_web_trf安装 en_core_web_sm 语言包非常的慢，经常会安装失败，这里我们可以离线安装。由于en_core_web_sm 对 spacy 的版本有较强的依赖性，你可以使用 pip show spacy 命令在终端查看你的版本，可以看到我的是 3.7.5 版本的 spacy。
![image](https://github.com/user-attachments/assets/13ee7ad7-20da-4cd1-b065-3999ff55e97e)

2. 数据预处理：
  * 定义 tokenizer：
      - 使用spacy对英文进行分词
      - 使用jieba对中文进行分词
        
            en_tokenizer = get_tokenizer('spacy', language='en_core_web_trf')
            zh_tokenizer = lambda x: list(jieba.cut(x))
        
  * 读取数据：
    
            def read_data(file_path: str) -> List[str]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f]

  * 数据预处理：

            def preprocess_data(en_data: List[str], zh_data: List[str]) -> List[Tuple[List[str], List[str]]]:
                processed_data = []
                for en, zh in zip(en_data, zh_data):
                    en_tokens = en_tokenizer(en.lower())[:MAX_LENGTH]
                    zh_tokens = zh_tokenizer(zh)[:MAX_LENGTH]
                    if en_tokens and zh_tokens:  # 确保两个序列都不为空
                        processed_data.append((en_tokens, zh_tokens))
                return processed_data

  * 构建词汇表和词向量：
    - 从训练数据中构建词汇表，并为每个词分配唯一索引。
    - 使用预训练或自训练词向量。

           def build_vocab(data: List[Tuple[List[str], List[str]]]):
                en_vocab = build_vocab_from_iterator(
                    (en for en, _ in data),
                    specials=['<unk>', '<pad>', '<bos>', '<eos>']
                )
                zh_vocab = build_vocab_from_iterator(
                    (zh for _, zh in data),
                    specials=['<unk>', '<pad>', '<bos>', '<eos>']
                )
                en_vocab.set_default_index(en_vocab['<unk>'])
                zh_vocab.set_default_index(zh_vocab['<unk>'])
                return en_vocab, zh_vocab
  * 定义数据集类：
    
        class TranslationDataset(Dataset):
            def __init__(self, data: List[Tuple[List[str], List[str]]], en_vocab, zh_vocab):
                self.data = data
                self.en_vocab = en_vocab
                self.zh_vocab = zh_vocab
        
            def __len__(self):
                return len(self.data)
        
            def __getitem__(self, idx):
                en, zh = self.data[idx]
                en_indices = [self.en_vocab['<bos>']] + [self.en_vocab[token] for token in en] + [self.en_vocab['<eos>']]
                zh_indices = [self.zh_vocab['<bos>']] + [self.zh_vocab[token] for token in zh] + [self.zh_vocab['<eos>']]
                return en_indices, zh_indices

  * 数据加载函数：

        def load_data(train_path: str, dev_en_path: str, dev_zh_path: str, test_en_path: str):
            # 读取训练数据
            train_data = read_data(train_path)
            train_en, train_zh = zip(*(line.split('\t') for line in train_data))
            
            # 读取开发集和测试集
            dev_en = read_data(dev_en_path)
            dev_zh = read_data(dev_zh_path)
            test_en = read_data(test_en_path)
        
            # 预处理数据
            train_processed = preprocess_data(train_en, train_zh)
            dev_processed = preprocess_data(dev_en, dev_zh)
            test_processed = [(en_tokenizer(en.lower())[:MAX_LENGTH], []) for en in test_en if en.strip()]
        
            # 构建词汇表
            global en_vocab, zh_vocab
            en_vocab, zh_vocab = build_vocab(train_processed)
        
            # 创建数据集
            train_dataset = TranslationDataset(train_processed, en_vocab, zh_vocab)
            dev_dataset = TranslationDataset(dev_processed, en_vocab, zh_vocab)
            test_dataset = TranslationDataset(test_processed, en_vocab, zh_vocab)
            
            from torch.utils.data import Subset
        
            # 假设你有10000个样本，你只想用前1000个样本进行测试
            indices = list(range(N))
            train_dataset = Subset(train_dataset, indices)
        
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
            dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
        
            return train_loader, dev_loader, test_loader, en_vocab, zh_vocab


3. 模型训练
    - 编码器-解码器模型：
    
    编码器将源语言句子编码成向量，捕捉源语言句子的语义信息。
    解码器将向量解码成目标语言句子，生成目标语言单词。
    - GRU网络：
    
    编码器和解码器均使用GRU，GRU是RNN的一种变体，具有更少的参数和更好的性能。
    编码器输出所有时间步的隐藏状态序列，包含源语言句子的所有信息。
    解码器初始隐藏状态为编码器的最后一个隐藏状态，启动解码过程。
   - 注意力机制：
    
    在解码过程中，计算注意力权重，生成注意力上下文向量。
    结合解码器隐藏状态和注意力上下文向量，生成目标语言单词。
    注意力机制允许解码器在生成每个输出词时，关注编码器产生的所有中间状态，更好地利用源序列的信息。

**Encoder**

    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
            super().__init__()
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            
            self.embedding = nn.Embedding(input_dim, emb_dim)
            self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, src):
            embedded = self.dropout(self.embedding(src))
            outputs, hidden = self.gru(embedded)
            return outputs, hidden

**Attention**

    class Attention(nn.Module):
        def __init__(self, hid_dim):
            super().__init__()
            self.attn = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)
            
        def forward(self, hidden, encoder_outputs):
            batch_size = encoder_outputs.shape[0]
            src_len = encoder_outputs.shape[1]
            
            hidden = hidden.repeat(src_len, 1, 1).transpose(0, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            attention = self.v(energy).squeeze(2)
            return F.softmax(attention, dim=1)

**Decoder**

    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
            super().__init__()
            self.output_dim = output_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.attention = attention
            
            self.embedding = nn.Embedding(output_dim, emb_dim)
            self.gru = nn.GRU(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
            self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, input, hidden, encoder_outputs):
            input = input.unsqueeze(1)
            embedded = self.dropout(self.embedding(input))
            a = self.attention(hidden[-1:], encoder_outputs)
            a = a.unsqueeze(1)
            weighted = torch.bmm(a, encoder_outputs)
            rnn_input = torch.cat((embedded, weighted), dim=2)
            output, hidden = self.gru(rnn_input, hidden)
            embedded = embedded.squeeze(1)
            output = output.squeeze(1)
            weighted = weighted.squeeze(1)
            prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
            return prediction, hidden

**构建Seq2Seq模型**

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            
        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.output_dim
            
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
            encoder_outputs, hidden = self.encoder(src)
            
            input = trg[:, 0]
            
            for t in range(1, trg_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[:, t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input = trg[:, t] if teacher_force else top1
            
            return outputs

**初始化模型**

    def initialize_model(input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout, device):
        attn = Attention(hid_dim)
        enc = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
        dec = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout, attn)
        model = Seq2Seq(enc, dec, device).to(device)
        return model

4. 训练过程
   * 初始化优化器并定义训练函数
  
         def initialize_optimizer(model, learning_rate=0.001):
             return optim.Adam(model.parameters(), lr=learning_rate)
         def train(model, iterator, optimizer, criterion, clip):
             model.train()
             epoch_loss = 0
             for batch in iterator:
                src, trg = batch
                if src.numel() == 0 or trg.numel() == 0:
                    continue
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                optimizer.zero_grad()
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
                loss.backward()
                clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss += loss.item()
             return epoch_loss / len(iterator)
    * 评价函数

          def evaluate(model, iterator, criterion):
            model.eval()
            epoch_loss = 0
            with torch.no_grad():
                for batch in iterator:
                    src, trg = batch
                    if src.numel() == 0 or trg.numel() == 0:
                        continue
                    src, trg = src.to(DEVICE), trg.to(DEVICE)
                    output = model(src, trg, 0)
                    output_dim = output.shape[-1]
                    output = output[:, 1:].contiguous().view(-1, output_dim)
                    trg = trg[:, 1:].contiguous().view(-1)
                    loss = criterion(output, trg)
                    epoch_loss += loss.item()
            return epoch_loss / len(iterator)

    * 主训练循环
  
          def train_model(model, train_iterator, valid_iterator, optimizer, criterion, N_EPOCHS=10, CLIP=1, save_path='../model/best-model.pt'):
            best_valid_loss = float('inf')
            for epoch in range(N_EPOCHS):
                start_time = time.time()
                train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
                valid_loss = evaluate(model, valid_iterator, criterion)
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), save_path)
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

     
5. 对测试集进行翻译

        model.load_state_dict(torch.load('../model/best-model_test.pt'))
        save_dir = '../results/submit_task2.txt'
        with open(save_dir, 'w') as f:
            for batch in test_loader:
                src, _ = batch
                src = src.to(DEVICE)
                translated = translate_sentence(src[0], en_vocab, zh_vocab, model, DEVICE, max_length=50)
                results = "".join(translated)
                f.write(results + '\n')
            print(f"翻译完成，结果已保存到{save_dir}")

   
**训练过程详解**
在训练过程中，我们首先定义了优化器，使用Adam优化算法来优化模型的参数。训练函数train负责在每个批次的数据上进行前向传播、计算损失、反向传播和参数更新。我们通过clip_grad_norm_函数来防止梯度爆炸问题。

评估函数evaluate在验证集上进行模型的评估，不进行梯度更新，仅计算损失。主训练循环train_model负责在多个epoch上进行训练和评估，并在每个epoch结束时保存最佳模型。

**评价和翻译**
在开发集上进行评价时，我们加载最佳模型并计算BLEU分数，这是一种常用的机器翻译评价指标。在对测试集进行翻译时，我们同样加载最佳模型，并对每个测试样本进行翻译，将结果保存到文件中。

通过这些步骤，我们可以有效地训练和评估Seq2Seq模型，并在测试集上生成翻译结果。
