import torch
import torch.nn as nn
from model_configs import clip_config_set
from clip import CLIP

def run_causal_intervention():
    
    cfg = clip_config_set.clip_1_5B_engram
    model = CLIP(cfg).cuda()
    model.eval()

    target_layer = model.image_encoder.Encoder.engram_layer[0]

    dummy_weather_img = torch.randn(1, 3, 224, 224).cuda()
    dummy_weather_txt = torch.randint(0, 49408, (1, 77)).cuda()

    with torch.no_grad():
        out_img, out_txt, _, _ = model((dummy_weather_img, dummy_weather_txt))
        
        gate_values = target_layer.last_gate_values
        hash_ids = target_layer.last_hash_ids
        
        max_gate_idx = torch.argmax(gate_values).item()
        target_engram_id = hash_ids.flatten()[max_gate_idx].item()
        
        print(f"해시 ID [{target_engram_id}]번 슬롯이 가장 강하게 활성화")


    def ablation_hook(module, input, output):

        modified_output = output.clone()
        modified_output[0, max_gate_idx, :] = 0.0 
        return modified_output

    hook_handle = target_layer.register_forward_hook(ablation_hook)


    with torch.no_grad():
        ablated_out_img, _, _, _ = model((dummy_weather_img, dummy_weather_txt))

        sim = nn.functional.cosine_similarity(out_img, ablated_out_img)
        
        print(f"{target_engram_id}번 제거, 모델 판단 원본 대비 {sim.item()*100:.2f}% 수준 변형")
        if sim.item() < 0.5:
            print("good")

    hook_handle.remove()

if __name__ == "__main__":
    run_causal_intervention()
