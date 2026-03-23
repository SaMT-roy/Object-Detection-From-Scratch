import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        qkv = self.qkv(x)                      # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, Hd)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                         # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class MLP(nn.Module):  # SwiGLU
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=True)
        self.w_value = nn.Linear(dim, hidden_dim, bias=True)
        self.w_out = nn.Linear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))     # SiLU on gate
        value = self.w_value(x)
        x = gate * value
        x = self.w_out(x)
        x = self.drop(x)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, N, D)
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3.0,
        attn_dropout=0.0,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, attn_dropout, dropout
        )

        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-LN
        x = x + self.mlp(self.norm2(x))
        return x

# PURE VIT
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns:
            x: (B, N, D)
            grid_hw: (H_p, W_p)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0, "Image size must be divisible by patch size"

        H_p, W_p = H // P, W // P

        # (B, C, H, W) → (B, H_p, W_p, C, P, P)
        x = x.view(B, C, H_p, P, W_p, P)
        x = x.permute(0, 2, 4, 1, 3, 5)

        # (B, H_p, W_p, C, P, P) → (B, N, C*P*P)
        x = x.reshape(B, H_p * W_p, C * P * P)

        # Linear projection → embedding dim
        x = self.proj(x)  # (B, N, D)

        return x, (H_p, W_p)

class Encoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=3.0,
        dropout=0.0,
        attn_dropout=0.0,
        base_grid_size=(16, 16),  # reference size (e.g. 256/16)
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        num_base_patches = base_grid_size[0] * base_grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_base_patches, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                attn_dropout,
                dropout
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x, grid_hw):
        """
        x: (B, N, D)
        grid_hw: (H, W) patch grid
        """

        H0 = W0 = int(self.pos_embed.shape[1] ** 0.5)
        patch_pos = self.pos_embed.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)

        patch_pos = F.interpolate(
            patch_pos,
            size=grid_hw,
            mode="bicubic",
            align_corners=False
        )

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, patch_pos.shape[1])
        return patch_pos

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x, (H, W) = self.patch_embed(x)  # (B, N, D)

        pos_embed = self.interpolate_pos_encoding(x, (H, W))
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class PatchUnembedding(nn.Module):
    def __init__(self, patch_size, out_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} is not a perfect square"

        P = self.patch_size
        C = self.out_channels

        x = self.proj(x)                  # (B, N, C*P*P)
        x = x.view(B, H, W, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)   # (B, C, H, P, W, P)
        x = x.reshape(B, C, H * P, W * P)

        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        out_channels=3,
        embed_dim=192,
        depth=3,
        num_heads=3,
        mlp_ratio=3.0,
        dropout=0.0,
        attn_dropout=0.0,
        base_grid_size=(16, 16),
    ):
        super().__init__()

        num_base_patches = base_grid_size[0] * base_grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_base_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                attn_dropout,
                dropout
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)

        self.patch_unembed = PatchUnembedding(patch_size, out_channels, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x):
        """
        x: (B, N, D)
        """
        N = x.shape[1]
        H = W = int(N ** 0.5)
        assert H * W == N, "Token count must be a square"

        H0 = W0 = int(self.pos_embed.shape[1] ** 0.5)

        pos = self.pos_embed.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos,
            size=(H, W),
            mode="bicubic",
            align_corners=False
        )
        pos = pos.permute(0, 2, 3, 1).reshape(1, N, -1)
        return pos


    def forward(self, x):
        """
        x: (B, N, encoder_embed_dim)
        grid_hw: (H, W) patch grid
        """

        pos_embed = self.interpolate_pos_encoding(x)
        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = self.patch_unembed(x)
        
        return x
