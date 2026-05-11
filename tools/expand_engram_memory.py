import torch
import torch.nn as nn
from model_configs import clip_config_set
from clip import CLIP

def expand_memory_for_new_climate(model, new_slots_count=1):    
    vit_engram_layer = model.image_encoder.Encoder.engram_layer[0]
    
    old_embedding = vit_engram_layer.embedding
    old_vocab_size, emb_dim = old_embedding.weight.shape
    
    new_vocab_size = old_vocab_size + new_slots_count
    new_embedding = nn.Embedding(new_vocab_size, emb_dim).cuda()
    
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_embedding.weight.clone()
    
    vit_engram_layer.embedding = new_embedding
    
    # 그래디언트 차단 새로운 슬롯만 학습
    def hook_freeze_old_memory(grad):
        # 역전파 시 앞의 226개 행에 대한 기울기는 0으로
        grad_clone = grad.clone()
        grad_clone[:old_vocab_size] = 0.0
        return grad_clone
        
    new_embedding.weight.register_hook(hook_freeze_old_memory)
    
    return model

# 사용 예시
# model = expand_memory_for_new_climate(model, new_slots_count=1)
