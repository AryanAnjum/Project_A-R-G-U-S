"""ARGUS-SLT: part-wise ST-GCN pose encoder feeding an mBART text model.

The pose encoder runs one spatio-temporal graph network per body part (body,
hands with shared weights, face), downsamples time by 4, and mixes the parts
with a transformer. Its output is projected to the text model's width and
passed to mBART as `inputs_embeds`, the same way GFSLT-VLP feeds video
features. Two text models are supported:

    mbart50  pretrained facebook/mbart-large-50 (Bengali, English, German, ...)
             trained with LoRA or fully fine-tuned
    small    a small mBART trained from scratch, close in size to SignFormer-GCN
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBart50TokenizerFast, MBartConfig, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from sign_language.keypoints import PART_EDGES, PARTS

# Set ARGUS_SLT_MBART to a local folder to load mBART-50 offline.
MBART50 = os.getenv("ARGUS_SLT_MBART", "facebook/mbart-large-50")
LANG_CODES = {"bn": "bn_IN", "en": "en_XX", "de": "de_DE", "zh": "zh_CN"}
SPECIAL_IDS = (0, 1, 2, 3)  # <s>, <pad>, </s>, <unk> in the mBART-50 vocabulary
BOS, PAD, EOS, UNK = SPECIAL_IDS


def adjacency(edges, n):
    a = torch.eye(n)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each (frame, point).

    BatchNorm mixed zero-padded frames into its running statistics, so the
    encoder behaved differently in training and evaluation. This norm has no
    batch statistics and ignores padding.
    """

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):  # (B, C, T, V)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.1):
        super().__init__()
        self.register_buffer("A", A)
        self.B = nn.Parameter(torch.zeros_like(A))  # learned extra connections
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.norm = ChannelNorm(cout)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            ChannelNorm(cout),
            nn.Dropout(dropout),
        )
        if cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), ChannelNorm(cout))
        self.stride = stride

    def forward(self, x, mask):  # x: (B, C, T, V), mask: (B, T)
        y = torch.einsum("bctv,vw->bctw", self.gcn(x), self.A + self.B)
        y = self.tcn(self.norm(y))
        mask = mask[:, ::self.stride]
        # zero the padded frames so the next temporal conv cannot read them
        return F.relu(y + self.res(x)) * mask[:, None, :, None], mask


class PartEncoder(nn.Module):
    LAYERS = [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2)]

    def __init__(self, edges, n_points, in_ch=9, dropout=0.1):
        super().__init__()
        A = adjacency(edges, n_points)
        self.in_norm = nn.LayerNorm(in_ch * n_points)
        blocks, cin = [], in_ch
        for cout, stride in self.LAYERS:
            blocks.append(STGCNBlock(cin, cout, A, stride, dropout))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = cin

    def forward(self, x, mask):  # x: (B, T, V, C), mask: (B, T) -> (B, T/4, D)
        b, t, v, c = x.shape
        x = self.in_norm(x.reshape(b, t, v * c)) * mask[..., None]
        x = x.reshape(b, t, v, c).permute(0, 3, 1, 2)
        for block in self.blocks:
            x, mask = block(x, mask)
        return x.mean(-1).transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class PoseEncoder(nn.Module):
    TIME_STRIDE = 4

    def __init__(self, d_model=512, layers=2, heads=8, dropout=0.1, rgb_dim=0):
        super().__init__()
        n = {k: s.stop - s.start for k, s in PARTS.items()}
        self.body = PartEncoder(PART_EDGES["body"], n["body"], dropout=dropout)
        self.hand = PartEncoder(PART_EDGES["rhand"], n["rhand"], dropout=dropout)
        self.face = PartEncoder(PART_EDGES["face"], n["face"], dropout=dropout)
        width = self.body.out_dim + 2 * self.hand.out_dim + self.face.out_dim
        # optional I3D clip features, concatenated with the pose features per time step
        self.rgb = nn.Sequential(nn.LayerNorm(rgb_dim), nn.Dropout(dropout)) if rgb_dim else None
        width += rgb_dim
        self.fuse = nn.Sequential(nn.Linear(width, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        # final norm: with norm_first the residual stream is otherwise never normalised
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d_model),
                                                 enable_nested_tensor=False)
        self.d_model = d_model

    @classmethod
    def out_lengths(cls, lengths):
        for _, stride in PartEncoder.LAYERS:
            lengths = (lengths - 1) // stride + 1
        return lengths

    def forward(self, x, lengths, rgb=None):  # x: (B, T, V, 9), rgb: (B, T/4, rgb_dim)
        lengths = lengths.to(x.device)
        frame_mask = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).to(x.dtype)
        lhand = x[:, :, PARTS["lhand"]].clone()
        lhand[..., 0::3] = -lhand[..., 0::3]  # mirror x so both hands share one encoder
        feats = torch.cat([
            self.body(x[:, :, PARTS["body"]], frame_mask),
            self.hand(lhand, frame_mask),
            self.hand(x[:, :, PARTS["rhand"]], frame_mask),
            self.face(x[:, :, PARTS["face"]], frame_mask),
        ], -1)
        if self.rgb is not None:
            feats = torch.cat([feats, self.rgb(rgb.to(feats.dtype))], -1)
        out_len = self.out_lengths(lengths)
        mask = torch.arange(feats.shape[1], device=x.device)[None] < out_len[:, None]
        h = self.transformer(self.pos(self.fuse(feats)), src_key_padding_mask=~mask)
        return h, mask


class PrunedVocab:
    """Keeps only the mBART-50 tokens seen in the training text.

    mBART-50 has 250k tokens; a sign language corpus uses a few thousand. Cutting
    the embedding to those saves about 250M parameters and a lot of VRAM.
    """

    def __init__(self, tokenizer, keep_ids, lang_code):
        self.tok = tokenizer
        self.keep = sorted(set(keep_ids) | set(SPECIAL_IDS) | {tokenizer.convert_tokens_to_ids(lang_code)})
        self.old2new = {o: n for n, o in enumerate(self.keep)}
        self.lang_id = self.old2new[tokenizer.convert_tokens_to_ids(lang_code)]
        self.lang_code = lang_code

    @classmethod
    def build(cls, tokenizer, texts, lang_code):
        ids = set()
        for enc in tokenizer(text_target=list(texts))["input_ids"]:
            ids.update(enc)
        return cls(tokenizer, ids, lang_code)

    def __len__(self):
        return len(self.keep)

    def encode(self, texts, max_len=128):
        out = []
        for ids in self.tok(text_target=list(texts), truncation=True, max_length=max_len)["input_ids"]:
            out.append([self.old2new.get(i, UNK) for i in ids])
        return out

    def decode(self, batch_ids):
        old = [[self.keep[i] for i in ids if i < len(self.keep)] for ids in batch_ids]
        return [s.strip() for s in self.tok.batch_decode(old, skip_special_tokens=True)]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"keep": self.keep, "lang_code": self.lang_code}, f)

    @classmethod
    def load(cls, path, tokenizer):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(tokenizer, d["keep"], d["lang_code"])


def load_tokenizer(lang_code):
    return MBart50TokenizerFast.from_pretrained(MBART50, src_lang=lang_code, tgt_lang=lang_code)


def _prune_pretrained(model, keep):
    keep_t = torch.tensor(keep)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight[keep_t].clone()
        bias = model.final_logits_bias[:, keep_t].clone()
        model.resize_token_embeddings(len(keep))
        model.get_input_embeddings().weight.copy_(emb)
        model.final_logits_bias.copy_(bias)
    model.tie_weights()


def build_text_model(kind, vocab, finetune="lora", lora_r=16):
    if kind == "mbart50":
        model = MBartForConditionalGeneration.from_pretrained(MBART50)
        _prune_pretrained(model, vocab.keep)
        if finetune == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1,
                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
            model = inject_adapter_in_model(cfg, model)
            for name, p in model.named_parameters():
                p.requires_grad = "lora_" in name
    elif kind == "small":
        cfg = MBartConfig(
            vocab_size=len(vocab), d_model=256, encoder_layers=4, decoder_layers=3,
            encoder_attention_heads=4, decoder_attention_heads=4,
            encoder_ffn_dim=1024, decoder_ffn_dim=1024, dropout=0.1, attention_dropout=0.1,
            max_position_embeddings=1024, scale_embedding=True,
            pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, decoder_start_token_id=EOS,
        )
        model = MBartForConditionalGeneration(cfg)
    else:
        raise ValueError(f"unknown decoder {kind}")
    model.config.vocab_size = len(vocab)
    model.config.decoder_start_token_id = EOS
    model.generation_config.forced_bos_token_id = vocab.lang_id
    model.generation_config.decoder_start_token_id = EOS
    model.generation_config.max_length = None  # translate() sets max_new_tokens
    return model


class SignTranslator(nn.Module):
    """Pose encoder plus mBART.

    bridge="decoder" (default) hands the pose features straight to the mBART
    decoder's cross-attention and skips mBART's own text encoder. Feeding pose
    features through the frozen text encoder ("encoder", the GFSLT-VLP layout)
    left the decoder free to ignore the signing: on PHOENIX it produced the same
    stock sentence for most inputs.
    """

    def __init__(self, text_model, d_pose=512, pose_layers=2, dropout=0.1, bridge="decoder", rgb_dim=0):
        super().__init__()
        self.pose = PoseEncoder(d_pose, pose_layers, dropout=dropout, rgb_dim=rgb_dim)
        self.text = text_model
        self.bridge = bridge
        d_text = text_model.config.d_model
        self.proj = nn.Sequential(nn.Linear(d_pose, d_text), nn.LayerNorm(d_text))
        if bridge == "decoder":
            text_model.model.encoder.layers = nn.ModuleList()  # unused; frees its memory

    def encode(self, features, lengths, rgb=None):
        h, mask = self.pose(features, lengths, rgb)
        return self.proj(h), mask.long()

    def _text_inputs(self, embeds, mask):
        if self.bridge == "decoder":
            return {"encoder_outputs": BaseModelOutput(last_hidden_state=embeds), "attention_mask": mask}
        return {"inputs_embeds": embeds, "attention_mask": mask}

    def forward(self, features, lengths, labels, word_dropout=0.0, rgb=None):
        embeds, mask = self.encode(features, lengths, rgb)
        dec_in = shift_tokens_right(labels, PAD)
        if self.training and word_dropout > 0:
            # hide some previous words so the decoder has to look at the signing
            drop = (torch.rand(dec_in.shape, device=dec_in.device) < word_dropout) & (dec_in > UNK)
            dec_in = dec_in.masked_fill(drop, UNK)
            dec_in[:, :2] = shift_tokens_right(labels, PAD)[:, :2]  # keep </s> and the language token
        return self.text(**self._text_inputs(embeds, mask), decoder_input_ids=dec_in).logits

    @torch.no_grad()
    def translate(self, features, lengths, num_beams=5, max_new_tokens=80, length_penalty=1.0,
                  no_repeat_ngram_size=3, rgb=None):
        # no_repeat_ngram_size stops the "heute heute heute ..." loops a weak model falls into
        embeds, mask = self.encode(features, lengths, rgb)
        return self.text.generate(**self._text_inputs(embeds, mask), num_beams=num_beams,
                                  max_new_tokens=max_new_tokens, length_penalty=length_penalty,
                                  no_repeat_ngram_size=no_repeat_ngram_size)


def save_checkpoint(model, vocab, args, path):
    os.makedirs(path, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items()}
    torch.save({"model": state, "args": vars(args)}, os.path.join(path, "model.pt"))
    vocab.save(os.path.join(path, "vocab.json"))


def load_checkpoint(path, device="cpu"):
    """Rebuild a SignTranslator from a folder written by save_checkpoint."""
    ckpt = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=False)
    args = ckpt["args"]
    lang_code = LANG_CODES.get(args["lang"], args["lang"])
    tok = load_tokenizer(lang_code)
    vocab = PrunedVocab.load(os.path.join(path, "vocab.json"), tok)
    if args["decoder"] == "mbart50":
        # the pretrained weights are overwritten by the checkpoint, so only the shape matters
        text = MBartForConditionalGeneration(MBartConfig.from_pretrained(MBART50))
        text.resize_token_embeddings(len(vocab))
        if args["finetune"] == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            r = args["lora_r"]
            text = inject_adapter_in_model(LoraConfig(r=r, lora_alpha=2 * r, target_modules=[
                "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]), text)
        text.generation_config.forced_bos_token_id = vocab.lang_id
        text.generation_config.decoder_start_token_id = EOS
        text.generation_config.max_length = None
    else:
        text = build_text_model("small", vocab)
    model = SignTranslator(text, args["d_pose"], args["pose_layers"], bridge=args.get("bridge", "encoder"),
                           rgb_dim=args.get("rgb_dim", 0))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), vocab, args
