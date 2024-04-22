# 与 LLM 有关面试题整理

## Transformer

![Transformer](https://pic2.zhimg.com/v2-9812a09d34d18bb78979903711efcadb_r.jpg)



### 位置编码

#### 为什么Transformer需要Positional Encoding？

Transformer中无RNN的循环结构，无法感知一个句子中词语出现的先后顺序，而词语的位置是相当重要的
一个信息。故提出了位置编码（即Positional Encoding）来解决这个问题Transformer原版的位置编码
是正余弦函数编码，表达的是绝对位置信息，同时包含相对位置信息。但是经过后续的self-attention，
相对位置信息消失。

编码需要：

* 需要体现同一单词在不同位置的区别。
* 需要体现一定的先后次序，并且在一定范围内的编码差异不应该依赖于文本的长度，具有一定的不变性。
* 需要有值域的范围限制。

![位置编码](https://pic1.zhimg.com/v2-1aff0167b20fd284ee698f5cbd7150c0_r.jpg)

解释一下这个公式：

i代表d_model中的维度。pos代表token位置。

* 当 i=0 时，波长为2pai，当i最大时波长为10000*2pai。这样各种波长都有，能够在不同d上编码不同，使高维空间表示包含意义。
* 区分了奇偶数维度的函数形式。

![img](https://pic3.zhimg.com/v2-2bd7b550e95792468a61738450b1e878_r.jpg)

reference:

https://zhuanlan.zhihu.com/p/667752628

https://zhuanlan.zhihu.com/p/166244505


### Attention

#### Attention 公式

![img.png](https://pic3.zhimg.com/v2-f99f1f97894eb8a76b9be6b00b17b302_r.jpg)


#### 为什么需要Multi-Head Attention？

「模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身的位置」（虽然这符合常识）而可能忽略了其它位置；同时，使用多头注意力机制还能够给予注意力层的输出包含有不同子空间中的编码表示信息，从而增强模型的表达能力（类比CNN同时使用多个卷积核），多头注意力有助于网络捕捉到更丰富的特征/信息总结：①为了解决模型在对当前位置的信息进行编码时，会过度的将注意力集中于自身位置的问题；②一定程度上hhh越大整个模型的表达能力越强，越能提高模型对于注意力权重的合理分配

Reference：https://zhuanlan.zhihu.com/p/667752628


#### Self attention的时间复杂度

Self-Attention时间复杂度？【美团】Self-Attention时间复杂度： O(n2⋅d)O(n^2 \cdot d)O(n^2 \cdot d) ，这里，n是序列的长度，d是embedding的维度Self-Attention包括三个步骤：相似度计算，softmax和加权平均，它们分别的时间复杂度是：相似度计算可以看作大小为 (n,d) 和 (d,n) 的两个矩阵相乘：(n,d)∗(d,n)=O(n2⋅d)(n,d)*(d,n)=O(n^2 \cdot d)(n,d)*(d,n)=O(n^2 \cdot d) ，得到一个 (n,n) 的矩阵softmax就是直接计算了，时间复杂度为 O(n2)O(n^2)O(n^2)加权平均可以看作大小为 (n,n) 和 (n,d) 的两个矩阵相乘： (n,n)∗(n,d)=O(n2⋅d)(n,n)*(n,d)=O(n^2 \cdot d)(n,n)*(n,d)=O(n^2 \cdot d) ，得到一个(n,d) 的矩阵因此，Self-Attention的时间复杂度是 O(n2⋅d)O(n^2 \cdot d)O(n^2 \cdot d)


#### KV Cache 和 MQA

当前自回归模型，通常是Decoder Only结构。推理过程是一个token一个token生成的过程，每生成一个token，都会进行一次self-attention计算。这样的计算效率是很低的，因为每次计算都是重复的计算。为了提高计算效率，提出了KV Cache和MQA。
Refe:https://zhuanlan.zhihu.com/p/686149289

### Normalization

#### 为什么NLP不用BN？

* 一个batch中不同句子字符长度不等，虽然通过补0或截断后能达到相同的句子长度，对这样一个batch进行BatchNorm，反而会加大特征的方差
* NLP中一个batch中所有Tokens的第i维向量关联性不大，对它们进行BatchNorm会损失Tokens之间的差异性。而我们想保留不同Tokens之间的差异性，所以不在该维度进行Norm
* 有实验证明，将LayerNorm换为BatchNorm后，会使得训练中Batch的统计量以及统计量贡献的梯度不稳定（Batch的统计量就是Batch中样本的均值和方差，Batch的统计量不稳定也就是当前Batch的均值和运行到当前状态累加得到的均值间的差值有的大，有的小。方差也是类似的情况）
* LayerNorm对self-attention处理累加得到的向量进行归一化，降低其方差，加速收敛

#### RMSNorm

TODO







