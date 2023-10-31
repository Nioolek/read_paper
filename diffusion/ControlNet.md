# Adding Conditional Control to Text-to-Image Diffusion Models

## Introduction

Stable Diffusion 这类模型的优点是可以生成高质量的图像，但是缺点是生成的图像是随机的，无法控制生成的图像的内容。
本文提出了一种方法，可以在生成图像的同时，控制生成图像的内容。
ControlNet 的目的是：add spatial conditioning controls to large, pretrained textto-image diffusion models。（摘抄自原文）
用中文解释，意思就是在大型的预训练的文本到图像的扩散模型中，添加空间条件控制。
这个空间条件控制，可以包含：Sketch（素描）、Normal map（法线图）、Depth map（深度图）、Seg（掩码）、canny edge （canny边缘）、Human Pose（人体姿态）等。

### Stable Diffusion 的缺点

* 需要搭配大量的 prompt，才能控制生成内容，例如使用：4K，高清，detailed。
* 人的手指、头发等细节，经常会出现问题，例如生成6根手指，脸部畸形等。
* 不足以支持更多条件输入，例如：人体姿态、法线图、深度图等。


## Method

![img.png](https://github.com/Nioolek/read_paper/assets/40284075/5a19c732-4600-40e6-9d08-a61380d89ad1)
![img.png](https://github.com/Nioolek/read_paper/assets/40284075/a2d9779b-c834-4ed7-91eb-848f8fc872be)

* 作者使用了训练好的 Stable Diffusion 网络，并冻结了其全部的参数。
* 通过在 Stable Diffusion 网络上添加模块，来实现控制生成图像的内容。
* 作者使用了 zero convolution ，意思就是全部初始化权重为 0 的卷积层，保证在网络刚开始训练时，整体网络的输出就保持为原本 Stable Deffusion 的输出。
* 从结构图中可以看到，仅在 decoder 中，将 zero conv 结果与原本的输出相加，得到最终的输出。并没有在 encoder 中使用。
* 作者比较了 zero conv 和其他网络的效果，最终还是 zero conv 效果最好。

## 个人理解

该做法简单高效，使用少量数据集就可以实现生成内容的控制。
