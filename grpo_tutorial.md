# GRPO算法教程：基于强化学习的LLM思维链训练

## 1. 简介

GRPO（Generalized Reinforcement Policy Optimization）是一种基于PPO（Proximal Policy Optimization）的强化学习算法，用于优化大型语言模型（LLM）生成具有思维链（Chain of Thought）的回答。本教程基于`grpo_vllm_one.py`实现，详细介绍GRPO的工作原理和实现流程。

## 2. 系统架构

GRPO实现包含三个主要组件：

1. **参考模型服务器**：运行在独立GPU上，计算参考模型的log概率
2. **生成进程**：使用vLLM加速推理，生成候选答案并计算奖励
3. **训练进程**：运行GRPO算法优化模型参数


## 3. 核心组件详解

### 3.1 数据流处理

该实现使用二进制序列化进行高效数据传输：

```python
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
    
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
```

### 3.2 奖励函数

GRPO使用两个奖励函数：

1. **正确性奖励**：评估答案是否正确
```python
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) 
    if len(nums) == 0: return -1.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1
```

2. **格式奖励**：评估答案是否符合指定格式
```python
def reward_format(item, answer):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("<think>") + answer.count("</think>")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1
```

### 3.3 核心算法公式

#### 3.3.1 KL散度

KL散度用于限制当前策略与参考策略之间的差异，公式如下：

$$D_{KL}(\pi_{ref} \parallel \pi_{\theta}) = \mathbb{E}_{x \sim \pi_{ref}} \left[ \log \frac{\pi_{ref}(x)}{\pi_{\theta}(x)} \right]$$

在代码中的近似实现：

```python
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
```

##### 为什么在δ=0处进行泰勒展开？

在KL散度的泰勒展开中，选择在δ=0处展开有几个重要原因：

1. **模型接近性假设**：GRPO算法的核心思想是保持新策略π_θ不要偏离参考策略π_ref太远。δ=0表示两个模型的概率完全相同（π_ref(x) = π_θ(x)）。在训练过程中，我们期望两个模型的差异较小，所以在δ=0附近展开是合理的。

2. **算法稳定性**：PPO类算法使用信任区域方法，限制每次更新的步长。展开点δ=0对应于"不更新"的情况，而我们实际的更新会在其附近，这保证了展开的准确性。

3. **近似精度**：泰勒展开在展开点附近提供最准确的近似。由于KL惩罚项的目的是使π_θ不要偏离π_ref太远，所以实际训练中δ的值应该比较小，在δ=0附近展开可以获得更好的近似效果。

4. **数值稳定性**：在大型语言模型中，处理长序列和大量token时，数值稳定性至关重要。在δ=0处展开可以避免在δ较大时可能出现的指数爆炸或梯度消失问题。

5. **简化计算**：在δ=0处展开可以得到简洁的近似公式，计算效率更高，特别适合大规模的token级概率计算。

##### KL散度的泰勒展开推导

KL散度的泰勒展开近似是通过以下步骤推导的：

1. 定义 $\delta = \log \pi_{ref}(x) - \log \pi_{\theta}(x)$，即参考模型与当前模型对数概率之差

2. 将 $e^{\delta}$ 在 $\delta=0$ 处进行泰勒展开：

$$e^{\delta} = 1 + \delta + \frac{\delta^2}{2!} + \frac{\delta^3}{3!} + \ldots$$

3. 整理得到：

$$e^{\delta} - 1 - \delta = \frac{\delta^2}{2!} + \frac{\delta^3}{3!} + \ldots$$

4. 当 $\delta$ 较小时，高阶项可以忽略，所以有近似：

$$e^{\delta} - 1 - \delta \approx \frac{\delta^2}{2}$$

5. 因此，KL散度可以近似为：

$$D_{KL}(\pi_{ref} \parallel \pi_{\theta}) \approx \mathbb{E}_{x \sim \pi_{ref}} \left[e^{\delta} - 1 - \delta\right]$$

6. 代入 $\delta = \log \pi_{ref}(x) - \log \pi_{\theta}(x)$，得到：

$$D_{KL}(\pi_{ref} \parallel \pi_{\theta}) \approx \mathbb{E}_{x \sim \pi_{ref}} \left[e^{\log \pi_{ref}(x) - \log \pi_{\theta}(x)} - 1 - (\log \pi_{ref}(x) - \log \pi_{\theta}(x))\right]$$

$$D_{KL}(\pi_{ref} \parallel \pi_{\theta}) \approx \mathbb{E}_{x \sim \pi_{ref}} \left[\frac{\pi_{ref}(x)}{\pi_{\theta}(x)} - 1 - \log \frac{\pi_{ref}(x)}{\pi_{\theta}(x)}\right]$$

代码中，直接对每个token应用了这个近似，其中：
- `ref_per_token_logps` 是 $\log \pi_{ref}(x)$
- `per_token_logps` 是 $\log \pi_{\theta}(x)$

这种近似方法具有计算效率高、数值稳定性强的特点，特别适合于大规模语言模型的token级概率计算。

#### 3.3.2 策略比率与裁剪

策略比率是当前策略与生成数据时使用的旧策略之间的概率比值：

$$r(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{old}(a|s)}$$

在代码中实现为：

```python
ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
```

PPO的裁剪目标函数用于限制策略更新幅度：

$$L^{CLIP}(\theta) = \mathbb{E}_{t} \left[ \min(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t) \right]$$

其中$A_t$是优势函数，$\varepsilon$是裁剪参数（在代码中为`clip_param = 0.2`）。

代码实现：

```python
clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
```

### 3.4 GRPO核心损失计算

```python
def GRPO_step(batch):
    # 获取提示长度和输入
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    # 获取模型logits
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # 排除最后一个logit
    input_ids = inputs[:, 1:]   # 排除第一个输入ID
    
    # 计算当前模型的token log概率
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    
    # 获取参考模型的token log概率
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    
    # 计算KL散度（近似）
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    # 创建完成部分的掩码（排除padding）
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # 计算策略比率和裁剪（PPO算法核心）
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    
    # 最终损失：负策略目标 + beta * KL散度
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # 应用掩码并计算平均损失
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss
```

## 4. 训练流程

### 4.1 启动参考模型服务器

参考模型服务器加载模型并提供两个API端点：
- `/upload`：接收生成的数据
- `/get`：返回处理后的数据（包含参考模型的logprobs）

### 4.2 初始化训练

```python
# 初始化分布式训练
deepspeed.init_distributed()

# 启动生成进程
if dist.get_rank() == 0:
    print('\nSTART vLLM generation...\n')
    mp.set_start_method('spawn')
    Q = mp.Queue()
    p = mp.Process(target=gen_worker, args=(Q, gen_device))
    p.start()

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

# 初始化DeepSpeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                            model_parameters=model.parameters())
```

### 4.3 训练循环

```python
for step in progress:
    # 获取批次数据
    batch = get_batch()
    while batch is None:
        print('waiting for batch...'); time.sleep(1)
        batch = get_batch()

    # 计算GRPO损失
    loss = GRPO_step(batch)
    
    # 反向传播和优化
    engine.backward(loss)
    engine.step()

    # 更新生成模型
    if step % gen_update_steps == 0:
        dist.barrier()
        if dist.get_rank() == 0:
            state_dict = engine.module.state_dict()
            Q.put(state_dict)  # 发送更新后的模型参数到生成进程
        dist.barrier()

    # 保存检查点
    if step % save_steps == 0:
        # 保存模型逻辑
```

## 5. 关键超参数

- `beta = 0.04`：KL散度惩罚系数
- `clip_param = 0.2`：PPO裁剪参数
- `Q_batch_size = 5`：生成批次大小
- `num_pre_Q = 8`：每个问题生成的答案数量
- `train_batch_size = 8`：训练批次大小

## 6. 创新点

1. **分离参考模型和训练模型**：减少GPU内存使用
2. **使用vLLM加速生成**：提高生成效率
3. **双重奖励机制**：同时优化答案正确性和格式合规性
4. **动态批次处理**：通过队列和多进程实现异步训练

## 7. 结论

GRPO算法通过强化学习优化LLM的思维链输出，同时保持模型输出的格式规范。该实现使用分离的参考模型和异步生成，使得在有限计算资源下训练大型模型成为可能。 