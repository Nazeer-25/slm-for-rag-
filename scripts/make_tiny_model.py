"""Build a tiny, randomly-initialized DistilBERT + WordPiece tokenizer locally.

ONLY for offline smoke tests (CI, sandboxes without HuggingFace access). It proves
the training/inference code paths run; it is not a substitute for
distilbert-base-uncased.

Usage: python scripts/make_tiny_model.py --corpus data/processed_smoke/chunks.jsonl --out models/tiny-distilbert
"""
import argparse
import json

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

texts = [json.loads(line)["text"] for line in open(args.corpus, encoding="utf-8")]
tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tok.normalizer = normalizers.BertNormalizer(lowercase=True)
tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tok.train_from_iterator(texts, trainers.WordPieceTrainer(
    vocab_size=4000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                               sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)
fast.save_pretrained(args.out)
cfg = DistilBertConfig(vocab_size=fast.vocab_size, dim=128, hidden_dim=256, n_layers=2, n_heads=2,
                       max_position_embeddings=512, num_labels=2)
DistilBertForSequenceClassification(cfg).save_pretrained(args.out)
print(f"tiny model saved to {args.out} (vocab={fast.vocab_size})")
