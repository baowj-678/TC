# Transformer:truck:

### Positional Encoding

**Transformer**不存在类似**RNN**的递归或者前向计算过程，因此为了表示数据的位置信息，我们需要给位置进行编码。这里，**Transformer**直接在**词向量(input embedding)**上面算术地加上**位置编码**，编码方式如下：

$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$$

$$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$

其中，$pos$是词在句中的位置，$i$是编码的维度，$d_{model}$是**词向量**的维度。

代码如下：

~~~python
class PositionalEncoding(NN.Module):
    """ 位置编码 """
    def __init__(self, d_model, max_len=50000):
        """
        Param
        -----
        :d_model 词向量的维度
        :max_len 句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        pos_enc = torch.zeros(max_len, d_model)
        # (max_len, d_model)
        pos = torch.arange(0, max_len, 1).unsqueeze(1)
        # (max_len, 1)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        # (d_model/2)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        # (1, max_len, d_model)
    
    def forward(self, X:torch.tensor):
        """
        Param
        -----
        :X [batch_size, length, d_model]
        
        Return
        ------
        :X [batch_size, length, d_model]
        """
        X =  X + Variable(self.pos_enc[:, 0:X.shape[1]], requires_grad=False)
        return X
~~~



### Attention

**注意力**可以被描述为：将**查询向量(query)**作用于**(mapping)键值向量对(key-value pair)**得到一个输出。其输出是**值向量(value)**的**权重和(weighted sum)**，而该权重又是**查询向量(query)**与其对应的**键向量(key)**的计算结果。



### Scaled Dot-Product Attention

<img src=".md/p1.png" alt="scaled dot-product attention" style="zoom:50%;" />



其输入是维度为$d_k$的**查询向量**和**键向量**，以及维度为$d_v$的**值向量**，计算公式如下：

$$Attention(Q,K,V)=\mathbb{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

常用的**注意力机制**有**加性注意力(additive attention)**和**乘性注意力(dot-product attention)**。这里在**乘性注意力**的基础上增加了一个**放缩因子(scaling factor)**，即$\frac{1}{\sqrt{d_k}}$。**乘性注意力**在实践中更快、更节约空间。

~~~python
def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制
    Param
    -----
    :query (batch_size, seq_len, d_k)
    :key (batch_size, seq_len, d_k)
    :value (batch_size, seq_len, d_v)
    :mask (batch_size, seq_len, seq_len)
    :dropout function

    Return
    ------
    :output (batch_size, seq_len, d_v)
    :p_attn (batch_size, seq_len, d_k)
    """
    d_k = query.shape[-1]
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # (batch_size, seq_len, seq_len)
    if mask is not None:
        score = score.masked_fill(mask, -100)
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
~~~





### Multi-Head Attention

相比于单个的**注意力**，更好的是将**查询向量**、**键向量**和**值向量**用$h$个学习得到的**线性层(linear projection)**分别投影到$d_k$和$d_v$维度。在每组投影后的向量组，可以并行地执行**注意力机制**，得到$d_v$维的输出。这些输出的向量会被**组合(concetenated)**然后再次执行上述操作。

<img src=".md\p2.png" style="zoom: 80%;" />



**多头注意力(Multi-Head Attention)**让模型能够同时**注意到(attend)**不同**表征子空间(Representation Subspace)**的信息。

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_1, \dots,head_h)W^O where \ head_i=\mathrm{Attention}(QW_i^Q,KW_i^K,V_i^V)$$

其中，$W_i^Q\in \mathbb{R}^{d_{model}\times d_k},W_i^K\in \mathbb{R}^{d_{model}\times d_k},W_i^V\in \mathbb{R}^{d_{model}\times d_v},W_i^O\in \mathbb{R}^{h*d_{v}\times d_{model}}$

*Transformer模型中使用*$h=8$ ,$d_k=d_v=d_{model}/h=64$。

~~~python
class MultiHeadAttention(NN.Module):
    """ 多头注意力 """
    def __init__(self, h, d_model, dropout=0.1) -> None:
        """
        Param
        -----
        :h 头的个数
        :d_model 模型维度
        :dropout 
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.W_O = NN.Linear(self.d_k * h, d_model)
        self.W_Q = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.W_K = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.W_V = [NN.Linear(d_model, self.d_k) for i in range(h)]
        self.dropout = NN.Dropout(dropout)
        self.attn = None
    
    def forward(self, query, key, value, mask=None):
        """
        Param
        -----
        :query (batch_size, seq_len, d_model)
        :key (batch_size, seq_len, d_model)
        :value (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)
        
        Return
        ------
        :x (batch_size, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(dim=0)
            # (1, batch_size, seq_len, seq_len)
        batch_size = query.shape[0]
        query = [network(query) for network in self.W_Q]
        key = [network(key) for network in self.W_K]
        value = [network(value) for network in self.W_V]
        # (h, batch_size, seq_len, d_k/d_v)
        query = torch.stack(query)
        key = torch.stack(key)
        value = torch.stack(value)
        # (h, batch_size, seq_len, d_k/d_v)
        x, self.attn = attention(query, key, value, mask)
        # (h, batch_size, seq_len, d_v)
        x = x.permute([1, 2, 0, 3])
        # (batch_size, seq_len, n, d_v)
        x = x.reshape(shape=(batch_size, -1, self.h * self.d_k))
        x = self.W_O(x)
        x = self.dropout(x)
        return x
~~~



### Position-wise Feed-Forward Network

除了**子层(sub-layers)**，**编码器(encoder)**和**解码器(decoder)**的每一层都包含一个**全连接网络(fully connect feed-forward network)**。

$$\mathrm{FFN}(x)=\max(m0,xW_1+b_1)W_2+b_2$$

*输出输出的维度为*$d_{model}=512$，*内部的维度为*$d_{ff}=2048$。

~~~python
class PositionwiseFeedForward(NN.Module):
    def __init__(self, d_model, d_ff, dropout=0) -> None:
        """
        Param
        -----
        :d_model 模型(输入)维度
        :d_ff 内部参数维度
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = NN.Linear(d_model, d_ff)
        self.relu = NN.ReLU()
        self.w_2 = NN.Linear(d_ff, d_model)
        self.dropout = NN.Dropout(dropout)
    
    def forward(self, X):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        X = self.relu(self.w_1(X))
        X = self.dropout(X)
        X = self.w_2(X)
        return X
~~~





### Embedding

这里使用学习得到的**词向量库**将**输入字词(token)**和**输出字词**转成$d_{model}$维的**词向量**；并且使用学得的**线性层**和**softmax层**将输出转成预测下一个**字词(token)**的**概率分布**。在该模型中，输入输出的Embedding共享参数，在Embedding层，在原来权重上乘以$\sqrt{d_{model}}$

~~~python
class Embedding(NN.Module):
    def __init__(self, vocab_size, d_model) -> None:
        """
        Param
        -----
        :vocab_size 词典大小(int)
        :d_model 模型维度(int)
        """
        super(Embedding, self).__init__()
        self.embeddings = NN.Embedding(vocab_size, d_model)
        self.sqrt_d_model = math.sqrt(d_model)
    
    def forward(self, X):
        """
        词向量编码
        Param
        -----
        :X [torch.tensor](batch_size, max_seq_len)

        Return
        ------
        :embed [torch.tensor](batch_size, max_seq_len, d_model)
        """
        embed = self.embeddings(X) * self.sqrt_d_model
        return embed
~~~



### Add & Norm

**残差连接**加上**层归一化**。

$$\mathrm{LayerNorm(x+\mathrm{Sublayer}(x))}$$

~~~python
class AddNorm(NN.Module):
    """ 残差连接 """
    def __init__(self, seq_len, d_model, dropout=0.1) -> None:
        super(AddNorm, self).__init__()
        self.layernorm = NN.LayerNorm((seq_len, d_model))
        self.dropout = NN.Dropout(dropout)
    
    def forward(self, X, sub_X):
        sub_X = self.dropout(sub_X)
        X = X + sub_X
        X = self.layernorm(X)
        return X
~~~



### Encoder Layer

每一个编码层有两个子层，第一层是**多头自注意力机制(multi-head self-attention mechanism)**，第二层是**位置全连接前馈层(position-wise fully  connected feed-forward network)**，每个子层后都会加一个**残差连接(residual connection)**和**层规范化化(layer normalization)**。

<img src=".md\p3.png" alt="EncoderLayer" style="zoom:80%;" />

~~~python
class EncoderLayer(NN.Module):
    def __init__(self, seq_len, d_model, h, dropout, d_ff) -> None:
        """
        一个编码层
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm  = AddNorm(seq_len, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, X, mask):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        sub_X = self.self_attn(X, X, X, mask)
        X = self.addnorm(X, sub_X)
        sub_X = self.feed_forward(X)
        X = self.addnorm(X, sub_X)
        return X
~~~



### Decoder Layer

每个解码层有三个子层，除了编码层的两层外，加入了在**编码器输出**和**解码器**之间的**多头注意力**层，即图中**Multi-Head Attention**层。其中**query**来自解码器，**key、value**均来自编码器输出。同时在**自注意力层**，即**Masked Multi-Head Attention**层，修改了**mask**的方法使得第$i$位置的字词生成只能依赖于**小于$i$的位置**的字词。

<img src=".md\p4.png" alt="DecoderLayer" style="zoom:80%;" />

~~~python
class DecoderLayer(NN.Module):
    """ 解码器的子层 """
    def __init__(self, tgt_len, d_model, h, dropout, d_ff) -> None:
        """
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        :d_ff
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm_1  = AddNorm(tgt_len, d_model, dropout)
        self.src_attn = MultiHeadAttention(h, d_model, dropout)
        self.addnorm_2  = AddNorm(tgt_len, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.addnorm_3  = AddNorm(tgt_len, d_model, dropout)


    def forward(self, X, M, src_mask, tgt_mask):
        """
        Param
        -----
        :X 训练时的目标语句 (batch_size, tgt_len, d_model)
        :M encoder得到的数据 (batch_size, src_len, d_model)
        :src_mask 对XM进行attention时的mask (batch_size, src_len, src_len)
        :tgt_mask 生成目标语句的mask (batch_size, tgt_len, src_len)

        Return
        ------
        :X (batch_size, tgt_len, d_model)
        """
        sub_X = self.self_attn(X, X, X, tgt_mask)
        X = self.addnorm_1(X, sub_X)
        sub_X = self.src_attn(X, M, M, src_mask)
        X = self.addnorm_2(X, sub_X)
        sub_X = self.feed_forward(X)
        X = self.addnorm_3(X, sub_X)
        return X
~~~





### Encoder

**编码器(encoder)**由$N=6$层**Encoder Layer**叠加而成。

~~~python
class Encoder(NN.Module):
    def __init__(self, seq_len, d_model, h, dropout, d_ff, n_layer=6) -> None:
        """
        编码器
        Param
        -----
        :seq_len 句子长度
        :d_model 模型维度
        :h 头数
        :dropout
        :d_ff 前馈层的内部向量维度
        :n_layer 编码器包含的层数
        """
        super(Encoder, self).__init__()
        encoder_layers = [EncoderLayer(seq_len, d_model, h, dropout, d_ff) for i in range(n_layer)]
        self.encoder = NN.Sequential(
            *encoder_layers
        )
    
    def forward(self, X, mask):
        """
        Param
        -----
        :X (batch_size, seq_len, d_model)
        :mask (batch_size, seq_len, seq_len)

        Return
        ------
        :X (batch_size, seq_len, d_model)
        """
        return self.encoder(X, mask)
~~~



### Decoder

**解码器(decoder)**由$N=6$层**Decoder Layer**叠加而成。

~~~python
class Decoder(NN.Module):
    """ 解码器 """
    def __init__(self, tgt_len, d_model, h, dropout, d_ff, n_layers) -> None:
        super(Decoder, self).__init__()
        self.decoderlayers = [DecoderLayer(tgt_len, d_model, h, dropout, d_ff) for i in range(n_layers)]
    
    def forward(self, X, M, src_mask, tgt_mask):
        """
        Param
        -----
        :X 训练时的目标语句 (batch_size, tgt_len, d_model)
        :M encoder得到的数据 (batch_size, src_len, d_model)
        :src_mask 对XM进行attention时的mask (batch_size, src_len, src_len)
        :tgt_mask 生成目标语句的mask (batch_size, tgt_len, src_len)

        Return
        ------
        :X (batch_size, tgt_len, d_model)
        """
        for decoderlayer in self.decoderlayers:
            X = decoderlayer(X, M, src_mask, tgt_mask)
        return X
~~~



### mask

**生成编码的Attention的mask**，如下格式

`[[False, False,  True,  True,  True],
      [False, False,  True,  True,  True],
      [ True,  True,  True,  True,  True],
      [ True,  True,  True,  True,  True],
      [ True,  True,  True,  True,  True]],`

~~~python
def en_seq_mask(seq_len, max_len):
    """
    encoder的句子mask
    Param
    -----
    :seq_len (list) (batch_size)
    :max_len (int)

    Return
    ------
    :mask (torch.ByteTensor) (batch_size, max_len, max_len)
    """
    batch_size = len(seq_len)
    mask = torch.ones(size=(batch_size, max_len, max_len), dtype=torch.bool)
    for mask_i, length in zip(mask, seq_len):
        mask_i[0:length, 0:length] = torch.zeros((length, length), dtype=torch.bool)
    return mask

~~~



**生成上三角的Attention**，如下格式

`[[[False,  True,  True,  True,  True,  True,  True],
       [False, False,  True,  True,  True,  True,  True],
       [False, False, False,  True,  True,  True,  True],
       [False, False, False, False,  True,  True,  True],
       [False, False, False, False, False,  True,  True],
       [False, False, False, False, False, False,  True],
       [False, False, False, False, False, False, False]]]`

~~~python
def de_seq_mask(tgt_len):
    """
    生成上三角的mask

    Param
    -----
    :tgt_len 方阵边长 (int)

    Return
    ------
    :mask (1, tgt_len, tgt_len)
    """
    mask = torch.ones((1, tgt_len, tgt_len), dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return mask
~~~

