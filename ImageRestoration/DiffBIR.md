# DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

## Introduction
本文解决的是：leverages pretrained text-to-image diffusion models for blind image restoration problem。
翻译成中文：利用预训练的文本到图像扩散模型进行盲图像恢复问题。


### 名词解释
首先要解释下，什么是盲图像恢复问题（Blind Image Restoration）。
* Blind Image Restoration（BIR）：盲图像恢复，在没有任何先验知识的情况下，对图像进行去噪、去模糊、去雾等操作。当前相关研究较少。
* blind image super-resolution (BSR), zero-shot image restoration (ZIR) and blind face restoration (BFR)这三个任务，都属于 BIR 任务。这三个任务已经有一些相关研究。

### 问题定义
作者对于 BIR 任务的解释是：
The ultimate goal of BIR is to realize realistic image reconstruction on general images with general degradations. BIR does not only extend the boundary of classic image restoration tasks, but also has a wide practical application field


个人认为，BIR任务是有一定意义的，因为在真实场景下，图像退化问题是很综合，很复杂的，并不单纯只包含某一种噪声或者某一种问题，而是多种问题同时存在，所以需要一个模型，能够认识到当前图像存在的问题，并同时解决多种问题。


## Methods

作者提出：
* 使用两阶段网络，第一阶段名为 Restoration Module ，第二阶段做图像重建。
* 第一阶段的目标是使用恢复模块来消除大部分损坏，例如噪声或失真伪影。原文直接使用了 SwinIR 作为恢复模块。
* 经过第一阶段，丢失的局部纹理和粗/细的细节仍然不存在，然后利用稳定扩散来弥补信息丢失。

![DiffBIR](https://github.com/Nioolek/read_paper/assets/40284075/005b4b60-6525-4fcd-a1f9-a2c0205831fe)

### Restoration Module

第一阶段的目标是使用恢复模块来消除大部分损坏。

比较值得注意的：
* 网络结构直接使用 SwinIR 。
* 对输入使用 pixel unshuffle 下采样8倍。后续所有的 transformer 模块都是基于下采样8倍来做的。
* 得到结果后，再使用3次 nearest interpolation ，并且每次 interpolation 后使用一次卷积和 Leaky ReLU 激活函数。

这个模块其实没有太多说的。

### Diffusion Module

第一阶段虽然能恢复图像，但是存在 over-smoothed 问题，并且 far from the distribution of high-quality natural images。
所以作者在第二阶段，引入 Diffusion Model ，来恢复图像细节。

（下面内容需要先对[ControlNet](../diffusion/ControlNet.md)和Stable Diffusion有一定了解）

#### 网络结构
流程：

1、首先使用 VAE 将 
$$\I_{reg}$$
编码到 latent space。

2、得到 latent space 的特征后，经过 t 次 Denoiser 

3、再经过 Decoder 得到最终输出。

#### 细节：

1、 Denoiser 的网络结构是 Stable Diffusion 的网络结构，但是权重是固定的，不参与训练。

2、添加一个Parallel Module ，用来控制生成图像的内容。Parallel Module 的 encoder 和 middle block 网络结构与 Denosier 相同，权重也与 Denosier 相同，但是 Parallel Module 的权重是参与训练的。

3、随机噪声 Z_t 与 $$\varepsilon (I_{reg})$$ concat 在一起，作为 Parallel Module 的输入。这会导致第一层卷积的 channel 数增加，这部分参数使用 0 进行初始化。

4、训练良好的 VAE 能够将条件图像投影到与潜在变量相同的表示空间中。

5、Stable Diffusion 的提示模块中，输入要为空。也就是说该网络并没有使用文本作为提示模块。

6、根据消融实验，使用ControlNet会导致颜色偏移，而作者使用的这种LAControlNet改善此问题。作者的解释是：LAControlNet 使用的是latent space 特征，并且 ControlNet训练没有 regularization on color consistency during training 。

## Latent Image Guidance for Fidelity-Realness Trade-off

待补充

## 限制

* 文章中提到，尚未探索文本驱动的图像恢复的潜力。期待后续相关工作。
* 引入了Diffusion，所以需要50个采样步骤来做图像恢复，相比较以前直接一个网络就可以做图像恢复，这个方法的速度会慢很多。