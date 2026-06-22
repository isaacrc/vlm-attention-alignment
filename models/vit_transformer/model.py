"""
models_vit_transformer.py
--------------------------
ViT-B/16 encoder + Transformer decoder.

Encoder: ViT-B/16 → 14×14×512 (same as models_vit_lstm.py).
Decoder: Transformer decoder with cross-attention (same structure as models_cnn_transformer.py)
         but encoder_dim=512 (no enc_proj needed, or trivial identity).
"""

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_HIDDEN = 768
ENC_DIM    = 512


class Encoder(nn.Module):
    """ViT-B/16 encoder — identical to models_vit_lstm.Encoder."""

    def __init__(self, encoded_image_size=14, proj_dim=ENC_DIM):
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.proj_dim = proj_dim

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.conv_proj   = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token
        self.proj        = nn.Linear(VIT_HIDDEN, proj_dim)
        self.fine_tune(fine_tune=False)

    def forward(self, images):
        B = images.size(0)
        x   = self.conv_proj(images)
        x   = x.flatten(2).transpose(1, 2)                # (B, 196, 768)
        cls = self.class_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                  # (B, 197, 768)
        x   = self.encoder(x)                             # (B, 197, 768)
        x   = x[:, 1:, :]                                 # (B, 196, 768)
        x   = self.proj(x)                                # (B, 196, 512)
        x   = x.view(B, 14, 14, self.proj_dim)            # (B, 14, 14, 512)
        return x

    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = False
        if fine_tune:
            for block in list(self.encoder.layers)[6:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.ln.parameters():
                p.requires_grad = True
            for p in self.conv_proj.parameters():
                p.requires_grad = True
            self.class_token.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True


class TransformerDecoderWithAttention(nn.Module):
    """
    Transformer decoder — same as models_cnn_transformer.py but encoder_dim=512.
    enc_proj is a trivial identity (or small projection) since encoder already at embed_dim.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, nhead, ff_dim,
                 encoder_dim=ENC_DIM, max_cap_len=52, dropout=0.5):
        super().__init__()
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead      = nhead
        self.max_cap_len = max_cap_len
        self.encoder_dim = encoder_dim

        # If encoder_dim == embed_dim, this is a trivial linear (can learn rescaling)
        self.enc_proj = nn.Linear(encoder_dim, embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc   = nn.Embedding(max_cap_len + 1, embed_dim)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.fc      = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def _run_layers(self, tgt, memory, tgt_mask=None):
        out = tgt
        all_cross_attn = []
        for layer in self.layers:
            out2 = layer.self_attn(out, out, out, attn_mask=tgt_mask,
                                   need_weights=False)[0]
            out  = layer.norm1(out + layer.dropout1(out2))
            out2, cross_attn = layer.multihead_attn(
                out, memory, memory,
                need_weights=True, average_attn_weights=True,
            )
            all_cross_attn.append(cross_attn)
            out  = layer.norm2(out + layer.dropout2(out2))
            out2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
            out  = layer.norm3(out + layer.dropout3(out2))
        out = self.norm(out)
        return out, all_cross_attn

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        B       = encoder_out.size(0)
        enc_dim = encoder_out.size(-1)

        memory_raw = encoder_out.view(B, -1, enc_dim)               # (B, 196, enc_dim)
        memory     = self.enc_proj(memory_raw)                       # (B, 196, embed_dim)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(descending=True)
        memory           = memory[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        decode_lengths = (caption_lengths - 1).tolist()
        max_len        = max(decode_lengths)

        tgt_tokens = encoded_captions[:, :max_len]
        pos_ids    = torch.arange(max_len, device=device).unsqueeze(0)
        tgt        = self.embedding(tgt_tokens) + self.pos_enc(pos_ids)

        tgt_mask   = nn.Transformer.generate_square_subsequent_mask(max_len, device=device)
        out, all_cross_attn = self._run_layers(tgt, memory, tgt_mask=tgt_mask)

        predictions  = self.fc(self.dropout(out))                    # (B, max_len, vocab_size)
        layer_alphas = torch.stack(all_cross_attn, dim=0)            # (num_layers, B, max_len, 196)
        alphas       = layer_alphas[-1]                              # (B, max_len, 196)

        return predictions, encoded_captions, decode_lengths, alphas, layer_alphas, None, None, sort_ind
