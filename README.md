# 🎬 Movie Review Sentiment Analysis using Deep Learning & NLP

> **Course Project** | IEEE-format Final Report | Due: May 5

## Team Information
Himaja Pusunuru
Tirumala Rao Peddakota
Leena Reddy Daida

## Project Overview
This project builds an end-to-end **sentiment analysis pipeline** for movie reviews using two deep learning architectures:
1. **Bidirectional LSTM with Self-Attention** (custom-trained word embeddings)
2. **DistilBERT** (fine-tuned Transformer)

The models classify IMDB reviews as **positive** or **negative** and are evaluated using five metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

## Dataset
- **IMDB Movie Reviews Dataset** – 50,000 samples (perfectly balanced: 25k pos / 25k neg)
- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Split: 80% train / 10% validation / 10% test (stratified)


## ML Techniques Used
| Technique | Implementation |
|---|---|
| Text Preprocessing | Regex cleaning, HTML stripping, lowercasing |
| Tokenization | Custom vocab (LSTM) / WordPiece (BERT) |
| Deep Learning | Bi-LSTM with Attention, DistilBERT fine-tuning |
| Regularization | Dropout, Gradient clipping, Weight decay |
| Optimization | AdamW + Cosine Annealing LR scheduler |
| Evaluation | Accuracy, Precision, Recall, F1, AUC-ROC |
| Visualization | ROC curves, confusion matrices, metric comparison |



## Results Summary
| Metric | Bi-LSTM | DistilBERT |
|---|---|---|
| Accuracy | ~89% | ~93% |
| Precision | ~88% | ~93% |
| Recall | ~90% | ~93% |
| F1-Score | ~89% | ~93% |
| AUC-ROC | ~95% | ~98% |

*(Actual results may vary slightly based on hardware and random seed)*

## Key Findings
- DistilBERT outperforms Bi-LSTM on all metrics due to pre-trained contextual embeddings
- Bi-LSTM with attention is a strong lightweight alternative (~4x faster training)
- Both models handle negation and sarcasm reasonably well on the IMDB domain

## References
- Maas et al. (2011) – *Learning Word Vectors for Sentiment Analysis*
- Sanh et al. (2019) – *DistilBERT, a distilled version of BERT*
- Hochreiter & Schmidhuber (1997) – *Long Short-Term Memory*
- Vaswani et al. (2017) – *Attention Is All You Need*
