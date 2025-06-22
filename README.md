## 🧠 Overview | 项目简介

**Text2Motion-RL** is a reinforcement learning (RL) based framework designed for generating realistic 3D human motions from natural language instructions. Unlike traditional supervised learning models, our approach leverages the exploration ability of RL to optimize motion quality based on imitation, smoothness, and semantic rewards.

**Text2Motion-RL** 是一个基于强化学习的文本驱动人体动作生成框架，能根据自然语言指令生成高质量的3D动作。该方法不依赖全监督数据，而是通过策略优化直接对运动质量进行优化。

---

## 🚀 Features | 项目亮点

- 🎯 Text-conditioned motion generation
- ♻️ Multi-objective reward: imitation, smoothness, semantics
- 📡 PPO-based actor-critic architecture
- 🧩 Support for HumanML3D dataset
- 🎥 Training progress & motion visualization

---

## 📦 Project Structure | 项目结构

```bash
text2motion_rl/
├── train/                  # 训练脚本
│   └── train_ppo.py
├── envs/                   # 自定义环境
│   └── text2motion_env.py
├── models/                 # 模型架构
│   ├── actor_critic.py
│   └── policy_wrapper.py
├── rewards/                # 奖励函数
│   └── reward_functions.py
├── utils/                  # 工具函数
│   └── logger.py
├── data/                   # 特征数据
│   ├── image_features.npy
│   ├── text_features.npy
│   └── action_labels.npy
└── README.md
```

---

## 🧩 Model Architecture | 模型结构

```bash
Text Instruction → Policy π → Action Sequence
                        ↓
             [Reward from Environment]
         ↑     ↑         ↑
     imitation  smoothness  semantic
```

We use an actor-critic architecture where the policy outputs 263D continuous action vectors. Observations include previous actions and text features.

---

## 🛠️ Current Issues & Suggestions | 当前问题与优化建议

### ❗ 现有问题
- 训练耗时长：每轮训练耗时达 600s+，GPU 使用效率不高
- 奖励函数过慢：当前 reward 函数基于 NumPy 和 fastdtw，效率较低
- Value Loss 波动大：说明 critic 网络学习不稳定
- Motion 维度处理不一致：GT 维度有时为 195，与 action 不匹配
- buffer 数据未标准化，影响稳定性

### ✅ 优化建议
- ✅ 使用 `torch` 重写 reward 函数，替代 `numpy+fastdtw`
- ✅ 改为 `generated_motion = np.zeros([max_len, act_dim])` 提前分配内存
- ✅ 对 `obs`, `action`, `reward` 做标准化
- ✅ 微调 critic 学习率并引入 reward clipping
- ✅ 引入 motion encoder 做语义奖励（可选）

---

## 📚 Dataset | 数据集


HEAD
We use the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset, which provides text-action pairs with aligned 3D pose sequences.
=======


---

## 📈 Training Logs | 训练日志样例

```bash
Epoch 0053 | Avg Reward: -329.79 | Policy Loss: 0.6477 | Value Loss: 117.6077
Epoch 0054 | Avg Reward: -328.25 | Policy Loss: 0.7786 | Value Loss: 193.6854
```

---

## 📦 Dependencies | 依赖环境

- Python 3.8+
- PyTorch 1.10+
- NumPy, tqdm, fastdtw
- HumanML3D preprocessing

---

## 📧 Contact | 联系方式

Please contact `kkuo4682@gmail.com` for any questions about this project.

---

**Enjoy motion generation with RL! | 强化学习让动作更智能！**
