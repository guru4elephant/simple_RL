from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
gen_device = 0    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04       # KL散度惩罚系数
all_steps = 1000
Q_batch_size = 5
num_pre_Q = 8     # 每个问题生成的样本数 G
train_batch_size = 8
gen_update_steps = 16
save_steps = 200
compute_gen_logps = True
epsilon_low = 0.8 # 下界裁剪参数
epsilon_high = 1.2 # 上界裁剪参数
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def is_equivalent(output, answer):
    # 判断输出是否与答案等价
    # 这里实现一个简单版本，可以根据具体任务定制
    # 例如，可以用reward_correct的逻辑来判断数学答案是否正确
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    output_nums = re.findall(pattern, output) 
    answer_nums = re.findall(pattern, answer)
    
    if len(output_nums) == 0 or len(answer_nums) == 0:
        return False
    
    # 比较最后的数字是否相等
    output_last = output_nums[-1]
    answer_last = answer_nums[-1]
    
    try:
        return float(output_last) == float(answer_last)
    except:
        return output_last == answer_last

def DAPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    
    # 获取标准化的奖励作为优势函数
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    
    # 获取模型logits
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit
    input_ids = inputs[:, 1:]   # (B, L-1), exclude the first input ID
    
    # 计算当前模型的token log概率
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    
    # 获取参考模型的token log概率
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    
    # 计算KL散度（近似）
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    
    # 创建完成部分的掩码（排除padding）
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # 计算策略比率
    if 'gen_logps' in batch:
        # 计算每个token的策略比率 r_{i,t}(θ)
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        
        # 使用双边裁剪 clip(r_{i,t}(θ), 1-ε_low, 1+ε_high)
        clipped_ratio = torch.clamp(ratio, 1-epsilon_low, 1+epsilon_high)
        
        # 取min，实现DAPO的目标函数
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    
    # 添加KL散度惩罚
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # 应用掩码并计算平均损失
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [];  ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    from math_verify import parse, verify, ExprExtractionConfig
    def reward_correct(item, answer):
        pattern = r'\d+\.\d+|\d+/\d+|\d+'
        nums = re.findall(pattern, answer) 
        if len(nums) == 0: return -1.0
        lastnum = nums[-1]
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        return 1 if verify(ans, ground_truth) else -1
    def reward_format(item, answer):
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        
        # 计算奖励和有效答案的数量
        rewards = []
        correct_counts = []
        
        for i, inp in enumerate(inputs):
            curr_rewards = []
            correct_count = 0
            
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                # 计算正确性和格式奖励
                r_correct = reward_correct(inp, a)
                r_format = reward_format(inp, a)
                curr_rewards.append(r_correct + r_format)
                
                # 检查答案是否等价于正确答案
                if r_correct > 0:
                    correct_count += 1
            
            rewards.extend(curr_rewards)
            correct_counts.append(correct_count)
            
            # 打印出每个问题有多少个正确答案
            print(f"Question {i}: {correct_count}/{num_pre_Q} correct answers")
        
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids, correct_counts

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids, correct_counts = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        if it % 5 == 0: print('answers:', answers[0])

        # 处理每个问题
        for i, pp in enumerate(prompt_inputs):
            # 跳过完全没有正确答案或全是正确答案的情况
            # 满足 0 < |{o_i | is_equivalent(a, o_i)}| < G 的约束条件
            if correct_counts[i] == 0 or correct_counts[i] == num_pre_Q:
                print(f"Skipping question {i} with {correct_counts[i]}/{num_pre_Q} correct answers")
                continue
                
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            
            # 确保奖励有足够的方差
            if curr_rewards.max() - curr_rewards.min() < 1e-4: 
                continue

            if ref_server_ver == 'tensor':
                # 标准化奖励，实现Â_{i,t}计算
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                    if compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss = DAPO_step(batch)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./dapo_step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier() 