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

当前自回归模型，通常是Decoder Only结构。推理过程是一个token一个token生成的过程，每生成一个token，都会进行一次self-attention计算。
这样的计算效率是很低的，因为每次计算都是重复的计算。为了提高计算效率，提出了KV Cache和MQA。

KV Cache将前向推理中的K和V进行缓存。在后续的自回归推理时，就不再需要重新推理K和V，以达到节省算力的目的。

但是，由于multi head attention中，会有多个head，不同head有不同的KV，会产生很多的中间变量，需要很多的显存。为了节省显存，在K和V计算时，就只计算单个head。
虽然相比原本的multi head方法降低了参数，但同时也会导致精度降低，却能大量减少中间结果显存需求。，这个做法叫做MQA。


#### GQA

既然只使用KV只使用1个head效果不好，那么就可以进行折中，那就多弄几套KV，数量仍然还是比Q的头数少一点。按照group的形式进行分配。
权重初始化的时候，使用average pooling得到参数，然后进行少量训练得到。
llama2用的就是GQA，其技术报告中展示了详细对比结果。

![img](https://pic2.zhimg.com/80/v2-aa6302478f6dab8cf4b4cc400a406f79_1440w.webp)

Reference:https://zhuanlan.zhihu.com/p/686149289

### Normalization

#### 为什么NLP不用BN？

* 一个batch中不同句子字符长度不等，虽然通过补0或截断后能达到相同的句子长度，对这样一个batch进行BatchNorm，反而会加大特征的方差
* NLP中一个batch中所有Tokens的第i维向量关联性不大，对它们进行BatchNorm会损失Tokens之间的差异性。而我们想保留不同Tokens之间的差异性，所以不在该维度进行Norm
* 有实验证明，将LayerNorm换为BatchNorm后，会使得训练中Batch的统计量以及统计量贡献的梯度不稳定（Batch的统计量就是Batch中样本的均值和方差，Batch的统计量不稳定也就是当前Batch的均值和运行到当前状态累加得到的均值间的差值有的大，有的小。方差也是类似的情况）
* LayerNorm对self-attention处理累加得到的向量进行归一化，降低其方差，加速收敛

#### Norm方式

有post-norm和pre-norm两种方式。post-norm的方式是在add之后进行LN，pre-norm实在全连接层之前进行LN。
![img](https://pic2.zhimg.com/80/v2-6304fc2872b860b363f462382fc887b5_1440w.webp)

使用PreNorm的网络一般比较容易训练。但是对于深层网络学习的效果不太好。因为PreNorm比较偏重来自底层的恒等分支。恒等分支更容易训练。
![img](https://pic4.zhimg.com/80/v2-3b38d786fb72296c95c243f39aea3fd3_1440w.webp)

#### RMSNorm

RMSNorm是在Layer Norm之上的改进，它通过舍弃中心不变性来降低计算量。公式如下：
![rmsnorm](https://pic2.zhimg.com/80/v2-901e415a3c14040af0362ca551c9416d_720w.webp)

Reference:

https://spaces.ac.cn/archives/8620

https://zhuanlan.zhihu.com/p/657659526

### 激活函数

softmax、relu这些激活函数就不提。

#### GELU 和 Swish 激活函数

Swish和GELU激活函数都是为了解决ReLu的缺陷而提出的。ReLu在负数部分的输出为0，导致了梯度消失的问题。

Swish激活函数公式：
```
$$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$
```

GELU激活函数是由Hinton等人提出，近似模拟神经元的激活方式。GELU的公式为：

```
$$ \text{GELU}(x) = x \cdot \Phi(x) $$
```

其中，$ x $是输入，$ \Phi(x) $ 是标准正态分布的累积分布函数（CDF），用于计算每个单位以某个概率激活。GELU通过随机性使模型更具鲁棒性，因此能够提高模型的泛化能力。在计算上，GELU通常使用以下近似形式来简化计算：

```
$$ \text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)\right]\right) $$
```

Swish和GELU的区别
数学形式：Swish函数用Sigmoid函数乘以输入，而GELU用输入乘以正态分布的CDF。
复杂度：GELU函数的复杂度略高于Swish，因为正态分布的CDF计算或其近似公式比Sigmoid函数更复杂。
性能：在不同的任务和网络架构中，这两种激活函数的性能可能有所不同。一些实验表明GELU在某些NLP任务中表现得更好，特别是在Transformer模型中得到广泛应用，而Swish则可能在其他类型的网络中有优势，性能差异很大程度上依赖于具体的数据集和任务。
鲁棒性和泛化：理论上，由于GELU考虑了激活的随机性，可能提供更好的鲁棒性和泛化能力。

SiLU其实就是beta为1时的Swish激活函数。



### 网络

#### 为什么现在的LLM都是Decoder only的架构？

* 工程：模型大了，对工程能力挑战比较大。
* 推理上，decoder-only可以更好的使用kv cache。
* zero-shot表现，有论文表示decoder only的表现更好。
* 有很多基础理论研究。例如很多研究者已经摸索出了基于decoder only的scaling law和训练方法，后来者鉴于时间和计算成本，不愿意做出大的改动。
* 从科学理论上分析：

@苏剑林 苏神强调的注意力满秩的问题，双向attention的注意力矩阵容易退化为低秩状态，而causal attention的注意力矩阵是下三角矩阵，必然是满秩的，建模能力更强；

@yili 大佬强调的预训练任务难度问题，纯粹的decoder-only架构+next token predicition预训练，每个位置所能接触的信息比其他架构少，要预测下一个token难度更高，当模型足够大，数据足够多的时候，decoder-only模型学习通用表征的上限更高；

@mimimumu 大佬强调，上下文学习为decoder-only架构带来的更好的few-shot性能：prompt和demonstration的信息可以视为对模型参数的隐式微调[2]，decoder-only的架构相比encoder-decoder在in-context learning上会更有优势，因为prompt可以更加直接地作用于decoder每一层的参数，微调的信号更强；

多位大佬强调了一个很容易被忽视的属性，causal attention （就是decoder-only的单向attention）具有隐式的位置编码功能 [3]，打破了transformer的位置不变性，而带有双向attention的模型，如果不带位置编码，双向attention的部分token可以对换也不改变表示，对语序的区分能力天生较弱。

Reference:https://www.zhihu.com/question/588325646/answer/3357252612


### 微调方法

#### 微调
高效微调粗略分为三类：加额外参数 A + 选取一部分参数更新 S + 引入重参数化 R
![finetune](https://pic2.zhimg.com/80/v2-ed42c72dfe5b849dfeb5df142f270675_720w.webp)

Reference: https://zhuanlan.zhihu.com/p/627537421

论文：Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning



### LLM 幻觉

大模型的幻觉，其实是产生了感觉上是真实的，但实际是虚假的感知。

就例如：对离离原上草一岁一枯荣是谁写的？这个问题，大模型可能会回答是李白，但实际上是苏轼。李白这个人是客观存在的，会让人信以为真。

引用中的文章解释了内在幻觉和外在幻觉。内在幻觉是模型自身产生了错误的回复。例如：提示词中用的是男的他，到回复里就变成女的她，这就是模型的错误。

外在幻觉是模型产生了一些内容，但是无法验证其的真实性。例如：明天会下雨。

#### 产生幻觉的原因

* 数据驱动原因:训练数据中源与参考的不匹配可能导致幻觉，如数据对不对齐，导致生成不忠实的文本。
* 表示和解码的不完善:编码器理解能力的缺陷和解码器策略错误可能导致幻觉。解码器Q可能关注错误的输入部分，或使用增加幻觉风险的策略，例如基于采样的解码中的随机性。
* 参数知识偏见:预训练模型可能偏好其参数中的知识而非新输入，从而导致幻觉。

#### 解决幻觉的方法


Reference:

为什么大模型会「说胡话」？如何解决大模型的「幻觉」问题？ - 平凡的回答 - 知乎
https://www.zhihu.com/question/635776684/answer/3336439291


### 评价指标

Accuracy
TODO








