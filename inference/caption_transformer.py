"""
caption_transformer.py
-----------------------
Beam search for Transformer-decoder captioning models.

Returns:
  seq          : list[int]  — best generated token sequence
  alphas       : np.ndarray (T, 196)  — last-layer cross-attn per word step
  layer_alphas : np.ndarray (num_layers, T, 196)  — all layers per word step

Usage:
  seq, alphas, layer_alphas = caption_image_beam_search_transformer(
      encoder, decoder, image_path, word_map, beam_size=5
  )
"""

import torch
import torch.nn.functional as F
import numpy as np
from imageio import imread
from skimage.transform import resize
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_column_mask(ablation_pct, seq_len, device):
    """
    Returns additive attn_mask (seq_len, 196) with -1e9 at left-column positions.
    Pass to multihead_attn as attn_mask; left columns collapse to ~0 after softmax.
    ablation_pct=0.0 returns None (no mask).
    """
    if ablation_pct <= 0.0:
        return None
    n_cols = int(14 * ablation_pct)
    mask_2d = torch.zeros(14, 14, device=device)
    mask_2d[:, :n_cols] = -1e9
    return mask_2d.reshape(1, 196).expand(seq_len, -1)  # (seq_len, 196)


def _load_image(image_path, resize_to=256):
    img = imread(image_path)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    img = resize(img, (resize_to, resize_to))
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize(img).unsqueeze(0)


@torch.no_grad()
def _decode_step(decoder, memory, seqs, ablation_pct=0.0, target_layer=None):
    """
    Run one transformer decoder forward pass, collecting both head-averaged
    and per-head cross-attention at each layer.

    Args:
        decoder      : TransformerDecoderWithAttention
        memory       : (B, 196, embed_dim) — projected encoder features
        seqs         : (B, T) — current token sequences (including <start>)
        ablation_pct : float — fraction of left columns to ablate (0 = no ablation)
        target_layer : int or None — which decoder layer to apply mask to;
                       None = all layers, int (0-3) = that layer only

    Returns:
        log_probs         : (B, vocab_size)
        cross_attns_avg   : list[num_layers] of (B, T, 196) — head-averaged
        cross_attns_perhead: list[num_layers] of (B, nhead, T, 196) — per-head
    """
    B, T  = seqs.size()
    embed = decoder.embedding(seqs)
    pos   = torch.arange(T, device=device).unsqueeze(0)
    tgt   = embed + decoder.pos_enc(pos)
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=device)

    # Build column mask once — reused for any layer that needs it
    col_mask = _make_column_mask(ablation_pct, T, memory.device)  # (T, 196) or None

    out = tgt
    cross_attns_avg     = []
    cross_attns_perhead = []

    for layer_idx, layer in enumerate(decoder.layers):
        # self-attention
        out2 = layer.self_attn(out, out, out, attn_mask=tgt_mask,
                               need_weights=False)[0]
        out = layer.norm1(out + layer.dropout1(out2))

        # Determine whether to apply column mask to this layer
        apply_mask = (col_mask is not None) and (
            target_layer is None or target_layer == layer_idx
        )
        layer_attn_mask = col_mask if apply_mask else None

        # cross-attention — collect both averaged and per-head
        out2, attn_avg = layer.multihead_attn(
            out, memory, memory,
            attn_mask=layer_attn_mask,
            need_weights=True, average_attn_weights=True)   # (B, T, 196)
        _, attn_ph = layer.multihead_attn(
            out, memory, memory,
            attn_mask=layer_attn_mask,
            need_weights=True, average_attn_weights=False)  # (B, nhead, T, 196)

        cross_attns_avg.append(attn_avg)
        cross_attns_perhead.append(attn_ph)
        out = layer.norm2(out + layer.dropout2(out2))

        out2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
        out = layer.norm3(out + layer.dropout3(out2))

    out = decoder.norm(out)
    logits    = decoder.fc(out[:, -1, :])
    log_probs = F.log_softmax(logits, dim=1)

    return log_probs, cross_attns_avg, cross_attns_perhead


@torch.no_grad()
def caption_image_beam_search_transformer(encoder, decoder, image_path, word_map,
                                          beam_size=5, resize_to=256,
                                          ablation_pct=0.0, target_layer=None):
    """
    Beam search for Transformer-based captioning model.

    Returns:
        seq                 : list[int]
        alphas              : np.ndarray (T, 196)             — last-layer head-avg, one map per word
        layer_alphas        : np.ndarray (num_layers, T, 196) — all layers head-avg
        layer_alphas_perhead: np.ndarray (num_layers, nhead, T, 196) — per-head per-layer
    """
    k          = beam_size
    vocab_size = len(word_map)
    start_tok  = word_map['<start>']
    end_tok    = word_map['<end>']

    # Encode image
    image = _load_image(image_path, resize_to=resize_to)
    enc_out    = encoder(image)
    enc_dim    = enc_out.size(-1)
    memory_raw = enc_out.view(1, -1, enc_dim)               # (1, 196, enc_dim)
    memory     = decoder.enc_proj(memory_raw)               # (1, 196, embed_dim)
    if hasattr(decoder, 'enc_norm'):
        memory = decoder.enc_norm(memory)
    memory = memory.expand(k, -1, -1)                       # (k, 196, embed_dim)

    # Initial sequences: all are [<start>]
    seqs       = torch.LongTensor([[start_tok]] * k).to(device)  # (k, 1)
    top_scores = torch.zeros(k, 1).to(device)                    # (k, 1)

    # Running attention history tensors
    seqs_attn_tensor    = None   # (active_k, num_layers, T, 196)
    seqs_attn_ph_tensor = None   # (active_k, num_layers, nhead, T, 196)

    complete_seqs           = []
    complete_seqs_attn      = []
    complete_seqs_attn_ph   = []
    complete_seqs_scores    = []

    step = 1
    while True:
        log_probs, cross_attns_avg, cross_attns_ph = _decode_step(
            decoder, memory[:seqs.size(0)], seqs,
            ablation_pct=ablation_pct, target_layer=target_layer)
        # cross_attns_avg: list[num_layers] of (active_k, T, 196)
        # cross_attns_ph:  list[num_layers] of (active_k, nhead, T, 196)

        # Take attention at the last generated position
        step_attn = torch.stack(
            [ca[:, -1, :] for ca in cross_attns_avg], dim=1
        )  # (active_k, num_layers, 196)
        step_attn_ph = torch.stack(
            [ca[:, :, -1, :] for ca in cross_attns_ph], dim=1
        )  # (active_k, num_layers, nhead, 196)

        scores = top_scores.expand_as(log_probs) + log_probs

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(seqs.size(0), 0, True, True)

        prev_inds = (top_k_words // vocab_size).long()
        next_inds = (top_k_words % vocab_size).long()

        seqs = torch.cat([seqs[prev_inds], next_inds.unsqueeze(1)], dim=1)

        # Extend head-averaged attention history
        cur_avg = step_attn[prev_inds].unsqueeze(2)         # (active_k, num_layers, 1, 196)
        if seqs_attn_tensor is None:
            seqs_attn_tensor = cur_avg
        else:
            seqs_attn_tensor = torch.cat(
                [seqs_attn_tensor[prev_inds], cur_avg], dim=2
            )  # (active_k, num_layers, T, 196)

        # Extend per-head attention history
        cur_ph = step_attn_ph[prev_inds].unsqueeze(3)       # (active_k, num_layers, nhead, 1, 196)
        if seqs_attn_ph_tensor is None:
            seqs_attn_ph_tensor = cur_ph
        else:
            seqs_attn_ph_tensor = torch.cat(
                [seqs_attn_ph_tensor[prev_inds], cur_ph], dim=3
            )  # (active_k, num_layers, nhead, T, 196)

        incomplete = [i for i, w in enumerate(next_inds) if w != end_tok]
        complete   = list(set(range(len(next_inds))) - set(incomplete))

        if complete:
            complete_seqs.extend(seqs[complete].tolist())
            complete_seqs_attn.extend(seqs_attn_tensor[complete].cpu().tolist())
            complete_seqs_attn_ph.extend(seqs_attn_ph_tensor[complete].cpu().tolist())
            complete_seqs_scores.extend(top_k_scores[complete].tolist())

        k -= len(complete)
        if k == 0:
            break

        seqs                = seqs[incomplete]
        seqs_attn_tensor    = seqs_attn_tensor[incomplete]
        seqs_attn_ph_tensor = seqs_attn_ph_tensor[incomplete]
        memory              = memory[:len(incomplete)]
        top_scores          = top_k_scores[incomplete].unsqueeze(1)

        if step > 50:
            break
        step += 1

    if not complete_seqs:
        # Fallback: take best incomplete sequence
        complete_seqs         = seqs.tolist()
        complete_seqs_attn    = seqs_attn_tensor.cpu().tolist()
        complete_seqs_attn_ph = seqs_attn_ph_tensor.cpu().tolist()
        complete_seqs_scores  = top_scores.squeeze(1).tolist()

    best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    seq      = complete_seqs[best_idx]
    attn_arr = np.array(complete_seqs_attn[best_idx])     # (num_layers, T, 196)
    attn_ph  = np.array(complete_seqs_attn_ph[best_idx])  # (num_layers, nhead, T, 196)

    alphas              = attn_arr[-1]   # (T, 196)
    layer_alphas        = attn_arr       # (num_layers, T, 196)
    layer_alphas_perhead = attn_ph       # (num_layers, nhead, T, 196)

    return seq, alphas, layer_alphas, layer_alphas_perhead
