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

  
  * 构建词汇表和词向量：
    从训练数据中构建词汇表，并为每个词分配唯一索引。
    使用预训练或自训练词向量。
  * 序列截断和填充：
    限制输入序列长度。
    填充序列至相同长度。
  * 添加特殊标记：
    添加 <SOS> 和 <EOS> 标记。
    添加 <UNK> 标记。
  * 数据增强：
    随机替换或删除词。
    同义词替换。
  * 数据分割：
    划分训练集、验证集和测试集。

3. 模型训练
编码器-解码器模型：

编码器将源语言句子编码成向量，捕捉源语言句子的语义信息。
解码器将向量解码成目标语言句子，生成目标语言单词。
GRU网络：

编码器和解码器均使用GRU，GRU是RNN的一种变体，具有更少的参数和更好的性能。
编码器输出所有时间步的隐藏状态序列，包含源语言句子的所有信息。
解码器初始隐藏状态为编码器的最后一个隐藏状态，启动解码过程。
注意力机制：

在解码过程中，计算注意力权重，生成注意力上下文向量。
结合解码器隐藏状态和注意力上下文向量，生成目标语言单词。
注意力机制允许解码器在生成每个输出词时，关注编码器产生的所有中间状态，更好地利用源序列的信息。
4. 翻译质量评价
人工评价：

系统上线前进行人工评价，确保系统性能准确。
有参考答案的自动评价：

使用BLEU等自动评价方法，快速、便捷地评估机器翻译系统的性能。
BLEU通过计算生成文本与参考译文之间的相似度，衡量翻译质量。
无参考答案的自动评价：

估计译文质量，选择性使用机器翻译译文。
通过模型置信度和可能性估计，提前知道译文的质量。
