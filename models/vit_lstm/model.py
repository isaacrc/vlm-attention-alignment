"""
models_vit_lstm.py
------------------
ViT-B/16 encoder + LSTM decoder with Bahdanau attention.

Encoder: ViT-B/16 (pretrained ImageNet-1K).
         Extract 196 patch tokens (14x14 grid), project 768 → 512.
         Output: (B, 14, 14, 512)  — same spatial layout as ResNet variant.
         encoder_dim = 512.

Decoder: identical to rnn_model/final_models.py DecoderWithAttention,
         but with encoder_dim=512 instead of 2048.

Fine-tuning: freeze all ViT blocks for Phase 1.
             Unfreeze last 6 blocks (indices 6-11) + conv_proj + LN for Phase 2+.
"""

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_HIDDEN = 768     # ViT-B/16 hidden dimension
ENC_DIM    = 512     # projected encoder dim (matches decoder default)


class Encoder(nn.Module):
    """ViT-B/16 encoder, outputs 14×14×512 feature grid."""

    def __init__(self, encoded_image_size=14, proj_dim=ENC_DIM):
        super().__init__()
        self.enc_image_size = encoded_image_size
        self.proj_dim = proj_dim

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # ViT components we need
        self.conv_proj   = vit.conv_proj          # patch embedding Conv2d(3,768,16,16)
        self.encoder     = vit.encoder            # TransformerEncoder (12 blocks)
        self.class_token = vit.class_token        # (1, 1, 768)
        # Note: vit.encoder also contains positional embedding

        # Project patch features to proj_dim
        self.proj = nn.Linear(VIT_HIDDEN, proj_dim)

        self.fine_tune(fine_tune=False)           # start frozen

    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            (B, 14, 14, proj_dim)
        """
        B = images.size(0)

        # Patch embedding: (B, 768, 14, 14) → (B, 196, 768)
        x = self.conv_proj(images)                              # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)                       # (B, 196, 768)

        # Prepend class token
        cls = self.class_token.expand(B, -1, -1)               # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)                         # (B, 197, 768)

        # ViT transformer encoder (includes positional encoding internally)
        x = self.encoder(x)                                     # (B, 197, 768)

        # Discard class token, keep patch tokens
        x = x[:, 1:, :]                                        # (B, 196, 768)

        # Project
        x = self.proj(x)                                       # (B, 196, proj_dim)

        # Reshape to spatial grid
        x = x.view(B, 14, 14, self.proj_dim)                   # (B, 14, 14, proj_dim)
        return x

    def fine_tune(self, fine_tune=True):
        """Freeze all, then selectively unfreeze last 6 ViT blocks."""
        for p in self.parameters():
            p.requires_grad = False
        if fine_tune:
            # Unfreeze last 6 of 12 encoder blocks
            encoder_layers = self.encoder.layers   # nn.Sequential of EncoderBlock
            for block in list(encoder_layers)[6:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Also unfreeze the final layer norm, patch embedding, class token
            for p in self.encoder.ln.parameters():
                p.requires_grad = True
            for p in self.conv_proj.parameters():
                p.requires_grad = True
            self.class_token.requires_grad = True
            # Always keep projection trainable
        for p in self.proj.parameters():
            p.requires_grad = True


# ── LSTM decoder with Bahdanau attention ────────────────────────────────────
# Identical to rnn_model/final_models.py except encoder_dim default = 512.

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att    = nn.Linear(attention_dim, 1)
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)

    def left_field_mask(self, noise_percentage, grid_size=14, device=None):
        assert 0.0 <= noise_percentage <= 1.0
        n_cols_mask = int(grid_size * noise_percentage)
        mask_2d = torch.ones(grid_size, grid_size, device=device)
        mask_2d[:, :n_cols_mask] = 0.0
        return mask_2d.reshape(1, -1)  # (1, 196)

    def forward(self, encoder_out, decoder_hidden, add_noise=False, noise_percentage=0.5):
        att1  = self.encoder_att(encoder_out)                            # (B, 196, attn_dim)
        att2  = self.decoder_att(decoder_hidden)                         # (B, attn_dim)
        att   = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (B, 196)
        alpha = self.softmax(att)                                        # (B, 196)

        if add_noise:
            B, n_pix = alpha.size()
            if noise_percentage == 0.0:
                alpha = torch.full((B, n_pix), 1.0 / n_pix,
                                   device=alpha.device, dtype=alpha.dtype)
            else:
                eps          = 1e-3
                spatial_mask = self.left_field_mask(
                    noise_percentage, grid_size=14, device=alpha.device
                ).expand(B, -1)                                          # (B, 196)
                unmasked     = alpha * spatial_mask
                masked       = torch.full_like(alpha, eps) * (1 - spatial_mask)
                total_eps    = eps * (1 - spatial_mask).sum(dim=1, keepdim=True)
                remaining    = 1.0 - total_eps
                unmasked     = unmasked / (unmasked.sum(dim=1, keepdim=True) + 1e-8) * remaining
                alpha        = unmasked + masked

        awe = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)             # (B, enc_dim)
        return awe, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=ENC_DIM, dropout=0.5):
        super().__init__()
        self.encoder_dim   = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.decoder_dim   = decoder_dim
        self.vocab_size    = vocab_size
        self.dropout       = dropout

        self.attention  = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding  = nn.Embedding(vocab_size, embed_dim)
        self.dropout    = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h     = nn.Linear(encoder_dim, decoder_dim)
        self.init_c     = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta     = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid    = nn.Sigmoid()
        self.fc         = nn.Linear(decoder_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        h = self.init_h(mean_enc)
        c = self.init_c(mean_enc)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        B         = encoder_out.size(0)
        enc_dim   = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(B, -1, enc_dim)         # (B, 196, enc_dim)
        num_pixels  = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(descending=True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings    = self.embedding(encoded_captions)        # (B, max_cap_len, emb_dim)
        h, c          = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(B, max(decode_lengths), vocab_size).to(device)
        alphas      = torch.zeros(B, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_t = sum(l > t for l in decode_lengths)
            awe, alpha = self.attention(encoder_out[:batch_t], h[:batch_t])
            gate = self.sigmoid(self.f_beta(h[:batch_t]))
            awe  = gate * awe
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_t, t, :], awe], dim=1),
                (h[:batch_t], c[:batch_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_t, t, :] = preds
            alphas[:batch_t, t, :]      = alpha

        # Return None for layer_alphas (LSTM has no layers), None placeholders
        return predictions, encoded_captions, decode_lengths, alphas, None, None, None, sort_ind
