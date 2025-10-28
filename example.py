import os
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import distributed
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config

from dinfer.model import FusedOlmoeForCausalLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

m = "/data/public_checkpoints/LLaDa-MoE-7B-A1B-Instruct-fused"
tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)

device = torch.device(0)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12346'
distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
distributed.initialize_model_parallel(1, backend='nccl')
parallel_config = ParallelConfig(enable_expert_parallel = True)
with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
    model_config = AutoConfig.from_pretrained(m, trust_remote_code=True)
    model = FusedOlmoeForCausalLM(config=model_config).eval()
    model.load_weights(m, torch_dtype=torch.bfloat16)
    model = model.to(device)

decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892)
dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'))

prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"
m = [{"role": "user", "content": prompt}, ]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
res = dllm.generate(input_ids, gen_length=1024, block_length=64)
print(tokenizer.decode(res[0, input_ids.shape[1]:], skip_special_tokens=False))