import torch
import torch.nn as nn


class PrefixTuningModel(nn.Module):
    def __init__(self, tokenizer, seq2seq_model, device, mid_dim=800, preseqlen=200, prefix_dropout=0.2):
        super(PrefixTuningModel, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.seq2seq_model = seq2seq_model
        self.freeze_params()

        self.config = self.seq2seq_model.config

        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = mid_dim
        self.preseqlen = preseqlen
        self.prefix_dropout = prefix_dropout
        self.vocab_size = len(self.tokenizer)

        self.input_tokens = torch.randint(0, self.preseqlen, (self.preseqlen,)).long()

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        )

        self.dropout = nn.Dropout(self.prefix_dropout)

    def get_prompt(self, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(temp_control_dec)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp['decoder_prompt'] = {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool()
                                      }
            key_val_dec = past_key_values_dec[i]
            temp['cross_attention_prompt'] = {"prev_key": key_val_dec[0].contiguous(),
                                              "prev_value": key_val_dec[1].contiguous(),
                                              "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(
                                                  key_val_dec.device).bool()
                                              }
            key_val_enc = past_key_values_enc[i]
            temp['encoder_prompt'] = {"prev_key": key_val_enc[0].contiguous(),
                                      "prev_value": key_val_enc[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(
                                          key_val_enc.device).bool()
                                      }
            result.append(temp)

        return result

    def forward(self, **batch):
        bsz = batch['input_ids'].shape[0]
        past_prompt = self.get_prompt(bsz=bsz)
        outputs = self.seq2seq_model(
            **batch,
            past_prompt=past_prompt
        )
        return outputs

    def freeze_params(self):
        for par in self.seq2seq_model.parameters():
            par.requires_grad = False
