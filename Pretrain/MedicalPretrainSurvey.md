# 文章名：Medical Vision Language Pretraining: A survey

2023年12月11日挂到 arxiv 上的文章，还比较新鲜热乎。多模态预训练近期可谓非常火热，在CV、NLP中能带来非常多的提升，但是在医疗中也在使用，但是没有像MAE等等这样非常有名的工作。笔者所看到的一些工作，也仅仅是将CV中使用的一些内容，迁移到医疗中，并没有什么核心创新，也没有解决很多本质问题。

故看到这篇文章时，特意拜读，并进行总结。

整体总结来说：
* 医疗预训练方面，和自然图像差距还是很大，差距原因包括：缺少基准数据集、模态多、数据量少、数据隐私性、数据特性等。
* 这篇文章，适合一扫而过，无需细读。
* 我觉得没有讲到核心问题，例如总结不出什么样的任务适合什么样的预训练方法。

## Introduction

本文主要内容：
* 介绍多模态学习，并公式化
* 详解各种方法和体系结构
* 给出VLP(Visual Language Pretraining)常用的医疗数据集
* 利用VLP的方法，也就是下游任务如何使用
* 探讨 limitations 和 challenges

## 介绍多模态预训练学习

这个其实有一定背景知识，都会了解。
分为监督学习和自监督学习。他们分别的做法就不赘述。

### 目标函数

目前主要分为4类：masked prediction, contrastive learning, matching prediction, and hybrid objectives

#### Masked Prediction

参照 MLM(Masked Language Modeling) 和 MIM(Masked Image Modeling) ，遮住一部分数据，然后预测遮住的数据。
如下图 (a)。
其中可以引入各种模态，输入和输出都可以同时包含图像、文本等模态。

该方法特性：
* 可以学习到 low level information.
* 网络模型一般特定于模态，引入复杂性。
* 预训练阶段和下游阶段之间出现了轻微的域差距，预训练阶段需要掩码输入，而下游任务涉及未屏蔽的输入。
* 仅依靠掩码预测目标的方法缺乏下游零样本能力。
* 数据集中会有罕见病、罕见条件，这些在图像中很细微，可能会遇到准确理解和表示这些罕见事件的困难。
* 医学文本经常具有写作风格的缩写和变化，这给 MLM 在有效学习一致表示方面带来了潜在的困难。

![img.png](img.png)


#### Contrastive Learning

##### Global Contrastive Learning

做法是让配对样本相似度更接近，不配对样本相似度更远。
医学里一些工作和clip比较像，或基于clip。他们主要是做将clip adapt to medical domain。


IMITATE 这个方法，分为'Findings' section和'Impressions' section，前者主要描述图像内容，后者反映和总结报告，使用CICL损失函数，高级部分和'Findings'对齐，多层次视觉特征和'Impressions'对齐。

##### Global and Local Contrastive Learning

比较需要提到的是，病灶一般在医疗图像中是非常小的，而且病灶的位置是不确定的。而clip这种方法，通常是关注于全局，这会导致一些问题。

GLORIA (解读：https://zhuanlan.zhihu.com/p/423871243) 中，分为Local contrastive loss和Global contrastive loss。
主要讲解Global contrastive loss, 首先会计算W个词和M个图像子区域的相似度。并将每一个词，对于所有图像区域的相似度整体进行归一化。最终图像子区域特征由相似度加权得到。
局部对抗损失，训练目标是给定word最大化图像的后验概率，以及给定图像最大化word 的后验概率。具体详解可以查看知乎链接。

GLORIA 这类方法，能够对齐图像和文本的局部信息，有利于下游任务。
但同时，这类方法局限性也很明显，单独的单词缺乏适当的上下文，不能确保和视觉直接对应。例如，liver cancer，如果按照单个字提取特征，cancer就不知道部位，liver则不知道疾病，导致很多问题。


##### False negatives in global contrastive learning

对比学习容易出现假阴性，其中负样本可能属于与锚点相同的类别，即使来自不同的样本，也可以共享相似的语义。这个问题在医学领域更成问题，其中数据集通常是高度不平衡的，与常见疾病或健康样本相比，罕见病理数量非常少。因此，负样本可能属于与锚点相同的类别的可能性更高。此外，医学数据集本质上是多类的，每种情况涉及多个病理条件[49]，因此锚点和负样本可能共享一些异常条件。这些假阴性无意中将具有相似语义的数据的表示推开，产生次优的图像-文本表示。
样例工作：MedCLIP 使用 UMLS 构建了一个知识驱动的相似性标签，该标签软化了语义匹配损失。


#### Matching Prediction

这个方法是将图像和文本他们的特征进行match。如果图和文能够配对，就要输出高概率；否则输出低概率。


#### Hybrid

这个方法是将上述的方法进行结合，例如将对比学习和掩码预测结合，或者将匹配预测和掩码预测结合。


## 附加的一些方面

### 使用未配对数据来预训练

该方法，主要是使用一些方法，造出预训练的target，例如利用类标签生成文本报告，造出图像文本对。

同时，其他一些方法借助LLM生成文本报告，使用扩散模型生成图像和对应的放射学报告。

### Using temporal information during pretraining

例如，医学中，包含对历史的引用，例如“Chest X 射线显示治疗后巩固和支气管血管标记减少”。

### Use of multiple views of Chest X-ray

使用X光的多个视图，进行特征对齐

### Other visual modalities

这部分讲述了处理3D图片和视频的方法。
例如adapt 2D ViT来处理3D图片。SurgVLP将手术讲座分割成随机短片段，用ASR模型提取音频转录成文字，并用预训练方法将视频剪辑和句子对齐。

## 数据增强

### 图像

一般就是图像常见的增强手段。但是注意文本中如果有左肺情况，但是随机裁剪裁掉了左肺，就可能导致问题。

### 文本

主要用于的文本增强技术包括句子、随机句子采样 [25] 和反向翻译的混洗


网络结构不多介绍


### Fusion方法

分为无fusion、early fusion、late fusion。
无fusion计算简单，能轻易地用于下游下游单模态和跨模态任务，但是多模态任务就会受限，例如VQA需要同时输入文本和图像。

early fusion和late fusion，及前者在很早时候就将模态融合，后者在很晚时候才融合。


## 下游任务

### 图像分类
#### zero shot classification

下游任务可以做zero shot classification（类似clip）。
在标记数据在下游任务中受到限制的情况下，零样本分类非常有效。然而，单独使用掩码预测目标训练的VLP缺乏零镜头分类能力，不能直接用于下游任务。因此，通常需要涉及对比学习的额外预训练目标。

#### Linear Probing

只训练一个线性层，用于分类。

#### Fine-tuning

微调。

### 图像seg

#### Zero-shot segmentation

由于预训练和分割差距较大，还是很难做的。
BioViL使用attention weights来做。

#### Fine-tuned segmentation
无需介绍

### 图像检测

无需介绍

### 跨模态检索

#### Zero shot retrieval

基于对比学习训练的VLP，可以很便捷的用于zero shot retrieval。
使用ITM损失训练的作品也能够执行零镜头检索。例如，从ITM头获得的匹配分数可以用来对候选查询进行排序，并以零镜头的方式进行检索。
但是ITM会遇到计算复杂度的问题，因为需要计算所有图像和文本的相似度。

### Medical Report Generation

zero shot方法CXR-RePair，使用基于检索的方法，根据图像特征检索相近的元素，但是这种方法目测并不靠谱。
ClinicalBERT, MedViLL, BioVil-T等方法基于early fusion结构，微调预训练模型，以生成文字。

### VQA

一般框架是基于预训练好的模型，添加fusion模块，然后微调。


数据集部分跳过。

