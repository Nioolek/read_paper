# Sigmoid Loss for Language Image Pre-Training

论文链接：https://arxiv.org/ftp/arxiv/papers/2303/2303.15343.pdf

当前在 Llava 等 VLM 模型中，会使用CLIP和SigLIP模型，作为图片特征提取器，且被证明在VLM中，这两个网络表现较好。
因为简单阅读此文章。

## 摘要

CLIP模型需要在相似度对比的时候，全局进行softmax计算，这样会导致batchsize特别大的时候很占用显存，无法使用特别大的batchsize。
SigLIP中使用Sigmoid Loss，可以在全局计算相似度，而不需要全局softmax。

SigLIP 使用4个TPUv4训练，仅训练两天,达到84.5%的zero-shot ImageNet准确率。

