import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_engram_source(model, image_tensor):
  print("엔그램 발현 위치 역추적")
  model.eval()
  image_tensor.requires_grad_()
    
  # 타겟 레이어
  target_layer = model.image_encoder.Encoder.engram_layer[0]
    
  _ = model((image_tensor, dummy_text))
  target_engram_id = 42 # 일단 가정하고 시작
    
  gates = target_layer.last_gate_values # [Batch, Seq(Patches), Vocab]
    
  engram_activation = gates[:, :, target_engram_id].sum()
    
  model.zero_grad()
  engram_activation.backward()
    
  saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1)
  saliency = saliency.squeeze().cpu().numpy()
    
  plt.imshow(saliency, cmap='hot')
  plt.title(f"Spatial Source of Engram #{target_engram_id}")
  plt.axis('off')
  plt.show()
  # 붉을수록 핵심 위치
