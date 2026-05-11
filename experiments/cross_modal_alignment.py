import torch
import torch.nn.functional as F
from model_configs import clip_config_set
from clip import CLIP

def check_engram_alignment():
    cfg = clip_config_set.clip_150M_normal
    model = CLIP(cfg).cuda()
    model.eval()

    typhoon_img = torch.randn(1, 3, 224, 224).cuda()
    typhoon_txt = torch.randint(0, 49408, (1, 77)).cuda()

    with torch.no_grad():
        _ = model((typhoon_img, typhoon_txt))

        vit_layer = model.image_encoder.Encoder.engram_layer[0]
        tet_layer = model.text_encoder.Encoder.engram_layer[0]

        vit_gates = vit_layer.last_gate_values.mean(dim=1).squeeze() # [vocab_size]
        tet_gates = tet_layer.last_gate_values.mean(dim=1).squeeze() # 같음

        alignment_score = F.cosine_similarity(vit_gates.unsqueeze(0), tet_gates.unsqueeze(0))
        
        vit_top_engram = torch.argmax(vit_gates).item()
        tet_top_engram = torch.argmax(tet_gates).item()

        print(f"구조적 일치도: {alignment_score.item()*100:.2f}%")
        print(f"시각 기억 슬롯: {vit_top_engram}번")
        print(f"언어 기억 슬롯: {tet_top_engram}번")
        
        if vit_top_engram == tet_top_engram:
            print("sync ok")
        else:
            print("break")

if __name__ == "__main__":
    check_engram_alignment()
