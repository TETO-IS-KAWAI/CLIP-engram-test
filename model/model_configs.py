from Image_encoder import VitConfig
from text_encoder import TetConfig
from engram import EngramConfig
from dataclasses import dataclass

'''
clip(normal)
clip(+ mHC)
clip(normal + engram)
clip(mHC + engram)
--> small, base
'''

#clip config, experiment setting
@dataclass
class clip_config :
    vit_config:VitConfig
    vit_engram_config:EngramConfig

    tet_config:TetConfig
    tet_engram_config:EngramConfig

@dataclass
class clip_config_set :
    clip_150M_normal = clip_config(
        vit_config = VitConfig(
                in_channels = 3,    
                img_size = 224,
                patch_size = 16,
                emb_dim = 768,
                depth = 12,
                n_heads = 12,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = False,
                use_mhc = False,
                device = "cuda"
        ),
        vit_engram_config = None,
        tet_config = TetConfig(
                vocab_size = 49408,
                max_ctx_len = 77,
                emb_dim = 512,
                depth = 12,
                n_heads = 8,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = False,
                use_mhc = False,
                device = "cuda",
        ),
        tet_engram_config = None
    )
    clip_1_5B_moe = clip_config(
        vit_config = VitConfig(
                in_channels = 3,
                img_size = 224,
                patch_size = 16,
                emb_dim = 768,
                depth = 12,
                n_heads = 12,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = True,
                every_2=True,
                k=2,
                c=1.0,
                n_experts=34,
                use_mhc = False,
                device = "cuda",
        ),
        vit_engram_config = None,
        tet_config = TetConfig(
                vocab_size = 49408,
                max_ctx_len = 77,
                emb_dim = 512,
                depth = 12,
                n_heads = 8,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = True,
                every_2=True,
                k=2,
                c=1.0,
                n_experts=34,
                use_mhc = False,
                device = "cuda",
        ),
        tet_engram_config = None,
    )

    clip_1_5B_engram = clip_config(
        vit_config = VitConfig(
                in_channels = 3,
                img_size = 224,
                patch_size = 16,
                emb_dim = 768,
                depth = 12,
                n_heads = 12,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = True,
                every_2=True,
                k=2,
                c=1.0,
                n_experts=34,
                use_mhc = False,
                device = "cuda",
        ),
        vit_engram_config = EngramConfig(
            embd_d=768,
        ),
        tet_config = TetConfig(
                vocab_size = 49408,
                max_ctx_len = 77,
                emb_dim = 512,
                depth = 12,
                n_heads = 8,
                attn_dropout = 0.1,
                ffn_mul = 4,
                ffn_dropout = 0.1,
                use_moe = True,
                every_2=True,
                k=2,
                c=1.0,
                n_experts=34,
                use_mhc = False,
                device = "cuda",
        ),
        tet_engram_config = None,
    )
