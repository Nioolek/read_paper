# reinforcement learning algorithm 强化学习


## 分类
* 基于价值（Value-based）
* 基于策略（Policy-based）
* 基于策略和价值（Actor-Critic）
* 基于模型（Model-based）
* 其他

### 基于价值（Value-based）
* Q-learning
* Deep Q Network (DQN)


总结一下，在强化学习中，我们关注的有如下几点：

* 环境观测值/状态 State
* 动作选择策略 Policy
* 执行的动作/行为 Action
* 得到的奖励 Reward
* 下一个状态 S’

![DQN](https://pic1.zhimg.com/80/v2-ed869e5520e2bfd920ec1ebd1b16d358_720w.webp)
要注意，上面这张图里，网络预测的应该是Q值，而不是policy。

DQN里，需要网络预测Q值。应用的时候，我们会选择Q值最大的动作作为输出，这个动作就是我们要执行的动作。

### 基于策略（Policy-based）

* Policy Gradient

![policygradient](https://upload-images.jianshu.io/upload_images/4155986-0da7e5f276ec5aca.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

Value-based网络输出的是Q值，也就是对每个动作的评价。而Policy-based网络输出的是每个动作的概率，也就是策略。

Value-Based 方法

代表算法: Q-Learning, Deep Q-Network (DQN), SARSA, etc.

优点: - 直观简单: 基于值的方法直接学习每个动作的价值（如Q值），然后简单选择价值最高的动作。 - 稳定性: 理论上来讲，当学习率适当并且探索足够时，值函数可以收敛到最优解。 - 可解释性: 值函数直接反映了采取不同动作的价值，通常更容易调试和解释。

缺点: - 动作空间局限: 基于值的方法一般适用于有限的离散动作空间，对于连续动作空间，它们都面临着一定的挑战。 - 评估策略依赖: 在某些环境中，评价一个动作需要知道后续的动作和奖励，这要求策略至少是部分确定性的。 - 需要最大化动作的步骤: 每次选择动作时都需要遍历所有的动作来找出价值最高的一个，这在动作空间很大时可能会成为瓶颈。

Policy-Based 方法
代表算法: REINFORCE, Policy Gradients, Actor-Critic, Proximal Policy Optimization (PPO), etc.

优点: - 适合连续动作空间: 因为策略函数可以直接输出动作而不需要额外的最大化步骤，这使得它们非常适合处理连续动作的问题。 - 学习随机策略: 有助于探索，因为Policy-Based方法直接学习的是动作的概率分布。 - 更强的逼近能力: 策略梯度方法理论上可以学习到更复杂的策略，包括那些值方法难以表达的策略。

缺点: - 效率较低: 通常需要比值方法更多的样本来稳定学习，因为它们直接在策略空间中搜索而不是学习一个价值映射。 - 高方差: 奖励通常只在序列结束时给出，这可能导致高方差并影响学习速度。 - 收敛难度: 策略空间可能非常庞大且复杂，找到最优策略可能效率低下且更难收敛。

### Actor-Critic 方法 (结合两者的优点)
Actor-Critic 方法结合了Value-Based和Policy-Based的优点，其中Actor负责更新策略（Policy-Based），Critic负责评价这个策略的好坏（Value-Based）。这种结合方式既允许学习随机策略，又能利用值函数来减少训练过程中的方差并加快学习速度。Actor-Critic方法试图平衡两种方法的优点，并且在许多现实世界问题中非常有效。
