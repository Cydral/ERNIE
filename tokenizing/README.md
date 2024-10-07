# ERNIE Tokenizer Models

This repository contains pre-trained SentencePiece tokenizer models for the ERNIE project. The tokenizers are available in two variants: English-French (EN-FR) and European languages (EU).

## Repository Structure

ERNIE/tokenizing/<br>
├── en-fr/<br>
└── eu/<br>

## EN-FR Tokenizer

The `en-fr` directory contains reference files for the SentencePiece tokenizer trained on a mixed corpus of English and French texts.

### Features

- **Vocabulary Sizes**: 3k, 8k, 12k, 20k, 40k, 80k, and 100k
- **Training Data**: Approximately 4.5 million sentences of varying lengths, totaling 800 million characters
- **Primary Languages**: English and French
- **Configuration**: Unigram model, capable of handling unknown terms and potentially other languages

## EU Tokenizer

The `eu` directory contains tokenizer models trained on a corpus of European languages.

### Features

- **Languages Covered**: Bulgarian (bg), Czech (cs), Danish (da), German (de), Greek (el), English (en), Spanish (es), Estonian (et), French (fr), Hungarian (hu), Italian (it), Lithuanian (lt), Latvian (lv), Dutch (nl), Polish (pl), Romanian (ro), Slovak (sk), Slovenian (sl), and Swedish (sv)
- **Training Data**: Over 10 million sentences, encompassing nearly 3.5 billion characters
- **Default Model**: The EU 12k model is the default tokenizer used by ERNIE

## Usage

Both tokenizers use the SentencePiece algorithm with a Unigram configuration. This setup allows for:

1. Efficient handling of out-of-vocabulary words
2. Potential adaptation to languages not included in the training data
