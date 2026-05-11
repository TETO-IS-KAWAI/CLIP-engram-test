import torch
from torch import Tensor, nn
import numpy as np

from Image_encoder import VIT
from text_encoder import TET

from model_configs import clip_config, clip_config_set

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP(nn.Module) :
    def __init__(self, clip_cfg:clip_config):
        super().__init__()
        self.i_cfg = clip_cfg.vit_config
        self.t_cfg = clip_cfg.tet_config
        self.i_engram_cfg = clip_cfg.vit_engram_config
        self.t_engram_cfg = clip_cfg.tet_engram_config

        self.n_ctx = self.t_cfg.max_ctx_len

        self.image_encoder = VIT(self.i_cfg, self.i_engram_cfg)    
        self.text_encoder = TET(self.t_cfg, self.t_engram_cfg)

        self.ln_t = nn.LayerNorm(self.t_cfg.emb_dim)
        self.ln_i = nn.LayerNorm(self.i_cfg.emb_dim)
    
        self.text_proj = nn.Parameter(torch.empty(self.t_cfg.emb_dim, self.t_cfg.emb_dim))
        self.img_proj = nn.Parameter(torch.empty(self.i_cfg.emb_dim, self.t_cfg.emb_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.t_engram_cfg :
            self.engram_embedding = nn.ModuleList([
                nn.Embedding((self.i_engram_cfg.engram_vocab_size * len(range(self.i_engram_cfg.max_ngram - 1))) * 2, self.i_engram_cfg.engram_embd_d) for _ in self.i_engram_cfg.engram_layer_n
            ])
        else :
            self.engram_embedding = None

    @property
    def initialize_parameters(self) :
        nn.init.normal_(self.text_encoder.token_emb.embedding.weight, std=0.02)
        nn.init.normal_(self.text_encoder.token_emb.pos_emb, std=0.01)

        i_std = (self.i_cfg.emb_dim ** -0.5) * ((2 * self.i_cfg.emb_dim) ** -0.5)
        for m in self.image_encoder.parameters() :
            nn.init.normal_(m, std=i_std)

        t_std = (self.t_cfg.emb_dim ** -0.5) * ((2 * self.t_cfg.emb_dim) ** -0.5)
        for m in self.text_encoder.parameters() :
            nn.init.normal_(m, std=t_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.t_cfg.emb_dim, self.t_cfg.emb_dim)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text(self, text) :
        if self.i_cfg.use_moe :
            feat, aux_loss = self.text_encoder(text, self.engram_embedding)
        else :
            feat = self.text_encoder(text, self.engram_embedding)

        if self.t_cfg.use_mhc :
            feat = self.ln_t(feat)[:, :, 0, :]
        else :
            feat = self.ln_t(feat)
        output = feat[torch.arange(feat.shape[0]), text.argmax(dim=-1)] @ self.text_proj

        if self.i_cfg.use_moe :
            return output, aux_loss
        return output
        
    def encode_image(self, image) :
        if self.i_cfg.use_moe :
            feat, aux_loss = self.image_encoder(image, self.engram_embedding)
        else :
            feat = self.image_encoder(image, self.engram_embedding)

        if self.i_cfg.use_mhc :
            feat = self.ln_i(feat)[:, :, 0, :]
        else :
            feat = self.ln_i(feat)
        output = feat[:, 0, :] @ self.img_proj


        if self.i_cfg.use_moe :
            return output, aux_loss
        return output


    def forward_soft_routing(self, x, embedding:nn.Embedding, k=3):
        q = self.norm_q[0](x) # [Batch, Seq, Dim]
        all_engrams = embedding.weight # [Total_Slots, Dim]
        memory_k = self.norm_k[0](self.key_projs[0](all_engrams)) # [Total_Slots, Dim]
        memory_v = self.val_proj(all_engrams) # [Total_Slots, Dim]

        attn_scores = torch.matmul(q, memory_k.T) / math.sqrt(self.engram_embd_d)
        topk_scores, topk_indices = torch.topk(attn_scores, k=k, dim=-1)
        topk_gates = torch.softmax(topk_scores, dim=-1) # [Batch, Seq, K]
    
        selected_v = memory_v[topk_indices] # [Batch, Seq, K, Dim]
        v_gated = (selected_v * topk_gates.unsqueeze(-1)).sum(dim=2)
        return x + v_gated

if __name__ == "__main__" :
    import model_configs
    import torchinfo

    temp_model = CLIP(clip_cfg=clip_config_set.clip_150M_normal).to(device="cuda", dtype=torch.float32)
    test_img = torch.randn((100, 3,224, 224), device="cuda", dtype=torch.float32)
    test_text = torch.randint(0, 100, (100, 77), device="cuda", dtype=torch.int32)

    torchinfo.summary(temp_model, input_data=[(test_img, test_text)])
