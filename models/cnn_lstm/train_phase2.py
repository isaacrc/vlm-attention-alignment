import time
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from final_models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = 'data/splits'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
suffix = '_phase2'  # distinct suffix so phase 2 checkpoints don't overwrite phase 1

# Model parameters
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Training parameters
start_epoch = 0
epochs = 10  # 10 additional epochs for encoder fine-tuning
batch_size = 32
workers = 1
encoder_lr = 1e-4  # smaller LR for encoder (as per methods)
decoder_lr = 4e-4  # decoder LR held constant
grad_clip = 5.
alpha_c = 1.
best_loss = float('inf')  # phase 2 tracks validation LOSS (not BLEU), per methods section
print_freq = 100
fine_tune_encoder = True  # unfreeze encoder layers 2-4

# Load best checkpoint from phase 1
checkpoint = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq_final.pth.tar'


def main():
    global best_loss, checkpoint, start_epoch, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Load phase 1 checkpoint
    print('Loading phase 1 checkpoint...')
    ckpt = torch.load(checkpoint, map_location=str(device))
    encoder = ckpt['encoder']
    decoder = ckpt['decoder']
    # Use checkpoint's decoder optimizer to preserve any LR decay from phase 1
    decoder_optimizer = ckpt['decoder_optimizer']

    # Enable gradient flow through encoder layers 2-4 and create fresh encoder optimizer
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=encoder_lr
    )

    # Move to GPU
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('Starting Phase 2: encoder fine-tuning (layers 2-4)')
    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch}')

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_bleu4, recent_loss = validate(val_loader=val_loader,
                                             encoder=encoder,
                                             decoder=decoder,
                                             criterion=criterion)

        # Phase 2 improvement = lower validation loss (per methods section)
        is_best = recent_loss < best_loss
        best_loss = min(recent_loss, best_loss)
        if not is_best:
            print(f'\nNo loss improvement. Best so far: {best_loss:.4f}\n')
        else:
            print(f'\nNew best validation loss: {best_loss:.4f}\n')

        # Save rolling checkpoint + BEST (lowest loss) checkpoint
        save_checkpoint(data_name + suffix, epoch, 0, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

        # Save epoch-stamped copy on every loss improvement for longitudinal analysis
        if is_best:
            improvement_dir = 'improvement_models/pretrain_phase2'
            os.makedirs(improvement_dir, exist_ok=True)
            state = {
                'epoch': epoch,
                'epochs_since_improvement': 0,
                'bleu-4': recent_bleu4,
                'val_loss': recent_loss,
                'encoder': encoder,
                'decoder': decoder,
                'encoder_optimizer': encoder_optimizer,
                'decoder_optimizer': decoder_optimizer
            }
            imp_filename = os.path.join(improvement_dir,
                f'epoch{epoch:03d}_loss{recent_loss:.4f}_bleu{recent_bleu4:.4f}_{data_name + suffix}.pth.tar')
            torch.save(state, imp_filename)
            print(f'Saved improvement checkpoint: {imp_filename}')


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, h_state, c_state, emb_, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Returns (bleu4, avg_val_loss). Phase 2 uses loss for improvement detection.
    """
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, h_state, c_state, emb_, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                          i, len(val_loader),
                          batch_time=batch_time, loss=losses, top5=top5accs))

            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses, top5=top5accs, bleu=bleu4))

    return bleu4, losses.avg


if __name__ == '__main__':
    main()
