"""
models_cnn_transformer.py
--------------------------
ResNet-101 encoder + Transformer decoder for image captioning.

Encoder: identical to rnn_model/final_models.py (ResNet-101, 14x14x2048 output).
Decoder: Transformer decoder with cross-attention over 196 image patches.
         Teacher-forcing forward pass for training.
         Returns per-layer cross-attention weights (shape: num_layers x B x T x 196).

Attention map extraction:
  - alphas       : last-layer cross-attention, mean over caption steps → (196,)
  - layer_alphas : all-layer cross-attention, mean over caption steps → (num_layers, 196)

API compatible with existing training loop in train_social_bs32_final.py:
  scores, caps_sorted, decode_lengths, alphas, layer_alphas, None, None, sort_ind
  = decoder(encoder_out, encoded_captions, caption_lengths)
"""

import torch
from torch import nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """ResNet-101 encoder — identical to rnn_model/final_models.py."""

    def __init__(self, encoded_image_size=14):
        super().__init__()
        self.enc_image_size = encoded_image_size
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)                # (B, 2048, H/32, W/32)
        out = self.adaptive_pool(out)            # (B, 2048, 14, 14)
        out = out.permute(0, 2, 3, 1)            # (B, 14, 14, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class TransformerDecoderWithAttention(nn.Module):
    """
    Transformer decoder for image captioning with explicit cross-attention extraction.

    Training: teacher-forcing (parallel forward pass with causal mask).
    Inference: autoregressive (see caption_transformer.py).

    Returns per-layer cross-attention for every forward pass.
    """

    def __init__(self, vocab_size, embed_dim, num_layers, nhead, ff_dim,
                 encoder_dim=2048, max_cap_len=52, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.max_cap_len = max_cap_len
        self.encoder_dim = encoder_dim

        # Project encoder features to embed_dim + normalize (ResNet lacks ViT's final LayerNorm)
        self.enc_proj = nn.Linear(encoder_dim, embed_dim)
        self.enc_norm = nn.LayerNorm(embed_dim)

        # Word embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Embedding(max_cap_len + 1, embed_dim)

        # Transformer decoder layers (manual, to extract per-layer cross-attn)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,   # (B, T, D) convention
                norm_first=False,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def _run_layers(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Run all decoder layers, collecting cross-attention weights at each layer.

        Args:
            tgt    : (B, T, embed_dim)
            memory : (B, S=196, embed_dim)
            tgt_mask: (T, T) causal mask
        Returns:
            out            : (B, T, embed_dim)
            all_cross_attn : list of (B, T, S) tensors, one per layer
        """
        out = tgt
        all_cross_attn = []

        for layer in self.layers:
            # --- self-attention ---
            out2 = layer.self_attn(out, out, out, attn_mask=tgt_mask,
                                   need_weights=False)[0]
            out = layer.norm1(out + layer.dropout1(out2))

            # --- cross-attention (capture weights) ---
            out2, cross_attn = layer.multihead_attn(
                out, memory, memory,
                need_weights=True,
                average_attn_weights=True,   # average over heads → (B, T, S)
            )
            all_cross_attn.append(cross_attn)   # (B, T, 196)
            out = layer.norm2(out + layer.dropout2(out2))

            # --- feed-forward ---
            out2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
            out = layer.norm3(out + layer.dropout3(out2))

        out = self.norm(out)
        return out, all_cross_attn

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Teacher-forcing forward pass.

        Args:
            encoder_out      : (B, 14, 14, encoder_dim)
            encoded_captions : (B, max_cap_len)  — includes <start>...<end>
            caption_lengths  : (B, 1)

        Returns (compatible with existing training loop):
            predictions  : (B, max_decode_len, vocab_size)
            caps_sorted  : (B, max_cap_len)
            decode_lengths: list[int]
            alphas       : (B, max_decode_len, 196)  ← last-layer cross-attn
            layer_alphas : (num_layers, B, max_decode_len, 196)  ← all layers
            None, None   : placeholders for h_state, c_state (LSTM compat)
            sort_ind     : sort indices
        """
        B = encoder_out.size(0)
        enc_dim = encoder_out.size(-1)

        # Flatten encoder → project to embed_dim → memory for decoder
        memory_raw = encoder_out.view(B, -1, enc_dim)                    # (B, 196, enc_dim)
        memory = self.enc_norm(self.enc_proj(memory_raw))               # (B, 196, embed_dim)

        # Sort by caption length (descending) — required by pack_padded_sequence
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(descending=True)
        memory = memory[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        decode_lengths = (caption_lengths - 1).tolist()
        max_len = max(decode_lengths)

        # Teacher-forcing input: <start> w1 w2 ... w_{T-1}  (length = max_len)
        tgt_tokens = encoded_captions[:, :max_len]             # (B, max_len)
        pos_ids = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        tgt = self.embedding(tgt_tokens) + self.pos_enc(pos_ids)     # (B, max_len, embed_dim)

        # Causal mask — prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_len, device=device)

        # Run layers, collect per-layer cross-attention
        out, all_cross_attn = self._run_layers(tgt, memory, tgt_mask=tgt_mask)

        # Predictions
        predictions = self.fc(self.dropout(out))    # (B, max_len, vocab_size)

        # Stack layer attentions: (num_layers, B, max_len, 196)
        layer_alphas = torch.stack(all_cross_attn, dim=0)
        alphas = layer_alphas[-1]                   # last layer: (B, max_len, 196)

        return predictions, encoded_captions, decode_lengths, alphas, layer_alphas, None, None, sort_ind
