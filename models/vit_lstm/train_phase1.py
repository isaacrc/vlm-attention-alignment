"""
CNN+Transformer Phase 1: COCO pretrain, encoder frozen.
Starts from scratch. Saves improvement checkpoints to checkpoints/phase1/.
"""
import os, sys, time, json
import torch, torch.backends.cudnn as cudnn
import torch.optim, torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

ARCH_DIR  = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
RNN_DIR   = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
COND_DIR  = os.path.join(ARCH_DIR, 'ViT_LSTM')
sys.path.insert(0, RNN_DIR)
sys.path.insert(0, os.path.join(ARCH_DIR, 'models'))

from models_vit_lstm import Encoder, DecoderWithAttention as Decoder
from datasets import CaptionDataset
from utils import AverageMeter, adjust_learning_rate, clip_gradient, accuracy

data_folder  = os.path.join(RNN_DIR, 'data', 'splits')
data_name    = 'coco_5_cap_per_img_5_min_word_freq'
IMPROVE_DIR  = os.path.join(COND_DIR, 'checkpoints', 'phase1')
os.makedirs(IMPROVE_DIR, exist_ok=True)
LOG_FILE     = os.path.join(COND_DIR, 'logs', 'phase1_loss.txt')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Hyperparameters
emb_dim       = 512
decoder_dim   = 512   # ff_dim
nhead         = 8
num_layers    = 4
max_cap_len   = 52
dropout       = 0.5
epochs        = 20
batch_size    = 32
workers       = 4
decoder_lr    = 4e-4
grad_clip     = 5.
alpha_c       = 0.0   # disabled; tune later
print_freq    = 200
fine_tune_enc = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
best_bleu4 = 0.
epochs_since_improvement = 0


def main():
    global best_bleu4, epochs_since_improvement

    word_map_file = os.path.join(data_folder, f'WORDMAP_{data_name}.json')
    with open(word_map_file) as f:
        word_map = json.load(f)
    vocab_size = len(word_map)

    encoder = Encoder()
    encoder.fine_tune(fine_tune_enc)
    decoder = Decoder(
        attention_dim=emb_dim, embed_dim=emb_dim, decoder_dim=decoder_dim,
        vocab_size=vocab_size, encoder_dim=512, dropout=dropout
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
    encoder_optimizer = None  # encoder frozen in phase 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224), normalize])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    print(f'=== ViT+LSTM Phase 1 | vocab={vocab_size} | device={device} ===')
    with open(LOG_FILE, 'w') as f:
        f.write('epoch,split,loss,top5acc,bleu4\n')

    for epoch in range(epochs):
        global epochs_since_improvement, best_bleu4
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        train_loss, train_top5 = train(train_loader, encoder, decoder, criterion,
                                       decoder_optimizer, encoder_optimizer, epoch)
        recent_bleu4 = validate(val_loader, encoder, decoder, criterion, word_map, epoch)

        with open(LOG_FILE, 'a') as f:
            f.write(f'{epoch},train,{train_loss:.4f},{train_top5:.3f},\n')
            f.write(f'{epoch},val,,, {recent_bleu4:.4f}\n')

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f'Epochs since improvement: {epochs_since_improvement}')
        else:
            epochs_since_improvement = 0
            ckpt = {'epoch': epoch, 'bleu-4': recent_bleu4,
                    'encoder': encoder, 'decoder': decoder,
                    'encoder_optimizer': None,
                    'decoder_optimizer': decoder_optimizer}
            path = os.path.join(IMPROVE_DIR, f'epoch{epoch:03d}_bleu{recent_bleu4:.4f}.pth.tar')
            torch.save(ckpt, path)
            print(f'Saved: {path}')


def train(loader, encoder, decoder, criterion, dec_opt, enc_opt, epoch):
    decoder.train(); encoder.eval()
    losses = AverageMeter(); top5 = AverageMeter(); batch_t = AverageMeter()
    start = time.time()
    for i, (imgs, caps, caplens) in enumerate(loader):
        imgs = imgs.to(device); caps = caps.to(device); caplens = caplens.to(device)
        enc_out = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, layer_alphas, _, _, sort_ind = \
            decoder(enc_out, caps, caplens)
        targets = caps_sorted[:, 1:]
        scores  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(scores, targets)
        if alpha_c > 0 and alphas is not None:
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        dec_opt.zero_grad()
        if enc_opt: enc_opt.zero_grad()
        loss.backward()
        clip_gradient(dec_opt, grad_clip)
        if enc_opt: clip_gradient(enc_opt, grad_clip)
        dec_opt.step()
        if enc_opt: enc_opt.step()
        t5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5.update(t5, sum(decode_lengths))
        batch_t.update(time.time() - start); start = time.time()
        if i % print_freq == 0:
            print(f'Epoch [{epoch}][{i}/{len(loader)}] '
                  f'Loss {losses.val:.4f}({losses.avg:.4f}) '
                  f'Top5 {top5.val:.3f}({top5.avg:.3f})')
    return losses.avg, top5.avg


def validate(loader, encoder, decoder, criterion, word_map, epoch):
    decoder.eval(); encoder.eval()
    losses = AverageMeter(); references = []; hypotheses = []
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(loader):
            imgs = imgs.to(device); caps = caps.to(device)
            caplens = caplens.to(device); allcaps = allcaps.to(device)
            enc_out = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, _, _, _, sort_ind = \
                decoder(enc_out, caps, caplens)
            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            scores  = pack_padded_sequence(scores,  decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = criterion(scores, targets)
            losses.update(loss.item(), sum(decode_lengths))
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                references.append([[w for w in c if w not in
                                     {word_map['<start>'], word_map['<pad>']}]
                                    for c in img_caps])
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            hypotheses.extend([preds[j][:decode_lengths[j]] for j in range(len(preds))])
    bleu4 = corpus_bleu(references, hypotheses)
    print(f'\n* Val Loss {losses.avg:.3f}  BLEU-4 {bleu4:.4f}\n')
    return bleu4


if __name__ == '__main__':
    main()
