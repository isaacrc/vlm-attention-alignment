import torch
from torch import nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        #### ***** NOW USING THE MOST RECENT RESNET WEIGHTS FOR THE 'FINAL' MODELS
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)  # UPDATED pretrained resnet -- use this eventually
        #resnet = resnet101(weights=ResNet101_Weights.DEFAULT)


        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    Inputs: Previous hidden state + the image
    Ouptus: soft attention vector across entire image -- probability of what you should attend to
    You could interpret this entire process as computing the probability that a pixel is the place to look to generate the next word
    2) This Attention class essentially computes the attention weights between the encoded image features and the decoder's hidden
    state to generate a weighted context vector, crucial for caption generation in image captioning models.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images (14x14x2388? should be 14x14 features x channels)
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        The encoder takes in a weighted average across all pixels + the previously output word
        """
        super(Attention, self).__init__()
        # nn.Linear = take the input of sixe (14x14x233 and transform it to 400 Dimensions, for example. This is the attention space - a vec of 400)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image 
        """
        This creates a linear layer that maps encoded image features to attention space.
        """
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        """
        This is another linear layer that maps the decoder's output (hidden cell state) to the same attention space
        """
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        """
        This weights both input sources -- the encoder input and the decoder input to produce 
        """
        self.relu = nn.ReLU()
        ## squash all weights between 0 and 1. Each weight is a vector across the ENTIRE attention space
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        ##
        #self.add_noise = add_noise
        #self.noise_percentage = noise_percentage

    def left_field_mask(self, noise_percentage: float, grid_size: int = 14, device=None):
        """
        Returns a (1, grid_size * grid_size) mask
        masking the left portion of the visual field.
        """
        assert 0.0 <= noise_percentage <= 1.0

        # number of columns to mask
        n_cols_mask = int(grid_size * noise_percentage)

        # grid: (H, W)
        mask_2d = torch.ones(grid_size, grid_size, device=device)
        mask_2d[:, :n_cols_mask] = 0.0  # mask left columns

        # flatten to match alpha
        mask_1d = mask_2d.reshape(1, -1)

        return mask_1d


    def forward(self, encoder_out, decoder_hidden, add_noise=False, noise_percentage=0.5):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # CALUCLATE ATTN WEIGHTS
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        n_pix = att.size(1) ## (num_pixels)
        batch_size = att.size(0)
        
        #print('ATTN SHAPE',att.shape)
        #print('ATTN vec:', att[0,:5])


        #### HEMI SPATIAL NEGLECT STUFF #### 
        '''
        Add noise to a percentage of units in att if add_noise is True
        In version 1:  X units, from top to bottom, are set to 1 (active), and the rest are set to 0 (inactive). (strongest neglect condition)
        In version 2: X units from left to right are set to 0 (inactive), and the rest of attention remains intact. (most accurate neglect condition) 

        '''
        # VERSION 1
        '''
        if add_noise:
            noise_units = int(n_pix * noise_percentage ) ## total pixels multiplied by the number to be noised 
            mask = torch.zeros_like(att)  # Create a zero tensor with the same shape as att
            mask[:, :noise_units] = 1  # Set the first 'noise_units' elements to 1
            att = mask  # Replace att with the mask

        '''

        # SOFTMAX: squish weights between 0 and 1
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # VERSION 2: Apply left field mask (X units from left to right set to 0, rest remain)
        # This version destroys attention AND perception due to the zeroing on the masked side, but keeps attn ditribution on the unmasked side
        '''
        if add_noise:
            spatial_mask = self.left_field_mask(noise_percentage, grid_size=14, device=alpha.device)  # (1, 196)

            # apply mask
            alpha = alpha * spatial_mask

            # renormalize remaining mass
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        
        # VERSION 3: Apply left field mask (X units from left to right set to 0, turn on ALL OTHER UNITS )
        # This version zeros some mass and redistributes the rest, without referencing the original attention distribution
        if add_noise:
            spatial_mask = self.left_field_mask(noise_percentage, grid_size=14, device=alpha.device)  # (1, 196)

            # apply mask
            alpha_new = spatial_mask

            # renormalize remaining mass
            alpha_new = alpha_new / (alpha_new.sum(dim=1, keepdim=True) + 1e-8)
            alpha = alpha_new.expand(alpha.size(0), -1)  # (s, 196)
        
        # VERSION 4: Apply left field mask (X units from left to right set to 0, rest remain, then renormalize according to SOFTMAX)
        # this doesn't work because when attention is low on the right side, the neglect side receives much attention
        alpha_new = att
        if add_noise:
            spatial_mask = self.left_field_mask(noise_percentage, grid_size=14, device=alpha_new.device)  # (1, 196)

            # apply mask and add back the batch dimension
            alpha_new = alpha_new * spatial_mask
            alpha_new = alpha_new.expand(att.size(0), -1)  # (s, 196)
        '''
        alpha_new = alpha

        if add_noise:
            if noise_percentage == 0.0:
                # CONTROL CONDITION:
                # Uniform attention over all spatial locations
                alpha = torch.full(
                    (batch_size, n_pix),
                    1.0 / n_pix,
                    device=alpha_new.device,
                    dtype=alpha_new.dtype
                )
            else:
                eps = 1e-3  # very small but non-zero attention floor

                # spatial_mask: (1, n_pix), 1 = unmasked, 0 = masked
                spatial_mask = self.left_field_mask(
                    noise_percentage, grid_size=14, device=alpha_new.device
                )

                # expand mask to batch
                spatial_mask = spatial_mask.expand(alpha_new.size(0), -1)

                # split attention into masked / unmasked parts
                unmasked_alpha = alpha_new * spatial_mask
                masked_alpha   = alpha_new * (1 - spatial_mask)

                # assign small floor to masked regions
                masked_alpha = torch.full_like(masked_alpha, eps) * (1 - spatial_mask)

                # compute how much mass is left for unmasked regions
                total_eps_mass = eps * (1 - spatial_mask).sum(dim=1, keepdim=True)
                remaining_mass = 1.0 - total_eps_mass

                # renormalize unmasked regions to absorb remaining mass
                unmasked_sum = unmasked_alpha.sum(dim=1, keepdim=True) + 1e-8
                unmasked_alpha = unmasked_alpha / unmasked_sum * remaining_mass

                # recombine
                alpha = unmasked_alpha + masked_alpha
        
        
        ### BLEND LEARNED ALPHAS (WEIGHTS) AND IMAGE TOGETHER
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        '''
        encoder_out * alpha.unsqueeze(2): This performs an element-wise multiplication between the encoder's output and the attention weights, broadcasting the weights across the
        feature dimensions. Specifically, each feature vector at every spatial position in the encoder's output is scaled by the corresponding attention weight. The result is a 
        tensor with the same shape as encoder_out (i.e., (batch_size, num_pixels, encoder_dim)).
        
        .sum(dim=1): This operation sums the weighted feature vectors across the spatial dimension (i.e., across num_pixels). The result is a tensor of shape (batch_size,
        encoder_dim).
        '''
        
        '''NOTES
        att1 results from a linear transformation (encoder_att) applied to the encoded image features (encoder_out). 
        The shape becomes (batch_size, num_pixels, attention_dim), where num_pixels refers to the number of pixels in 
        the image features and attention_dim is the dimension of the attention space.
        
        att2 results from a linear transformation (decoder_att) applied to the decoder's hidden state (decoder_hidden). 
        The shape becomes (batch_size, attention_dim).
        
        att2.unsqueeze(1) adds a new dimension to att2 to make its shape (batch_size, 1, attention_dim) for broadcasting purposes.
        att1 + att2.unsqueeze(1) adds the transformed encoded image features with the transformed decoder hidden state, 
        effectively blending the two representations.
        self.full_att(...) is another linear transformation (full_att) applied to the ReLU output, reshaping it 
        into (batch_size, num_pixels, 1).
        
        attention_weighted_encoding applies the attention calculation to the original image
        '''

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    Inputs: Soft attention computation [prediction of where to attend] + previously generated word
    Ouputs: next word
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param add_noise: boolean, whether to add noise
        :param noise_percentage: percentage of units to add noise to
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, add_noise=False, noise_percentage=0.5):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1) ##***** wait i think pixels are just flattened 14,14,2048!

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist() # get lengths of N ground truth captions (i think)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        h_state = torch.zeros(batch_size, max(decode_lengths), self.decoder_dim).to(device)
        c_state = torch.zeros(batch_size, max(decode_lengths), self.decoder_dim).to(device)
        emb_ = torch.zeros(batch_size, max(decode_lengths), self.embed_dim).to(device)


        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            ## pass thru attention net ## 
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], # Forward pass: batched images, batched hidden states
                                                                h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding 
            ## pass thru lstm                
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim) -- pass in all embeddings up to this timestep + the current attention weighted encoding
                        
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            h_state[:batch_size_t, t, :] = h
            c_state[:batch_size_t, t, :] = c
            emb_[:batch_size_t, t, :] = embeddings[:batch_size_t, t, :]

        return predictions, encoded_captions, decode_lengths, alphas, h_state, c_state, emb_, sort_ind
