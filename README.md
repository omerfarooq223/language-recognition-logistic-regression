# Language Recognition using Multinomial Logistic Regression

A machine learning project that classifies audio recordings into 4 languages — **Urdu, English, Turkish, and Mixed** — using MFCC features and Multinomial Logistic Regression implemented from scratch.

---

## Overview

This project was built as part of a Machine Learning course (AI-372) at UMT. Audio data was self-recorded in 4 languages and used to train and evaluate a language recognition classifier.

The project includes two implementations:
- **From Scratch** — Multinomial Logistic Regression built using only NumPy
- **Scikit-learn** — Using sklearn's `LogisticRegression` for comparison

---

## Dataset

- **Languages:** Urdu, English, Turkish, Mixed (Urdu + English)
- **Recordings:** 400–600 `.wav` files recorded using Praat
- **Sampling Rate:** 16,000 Hz, Mono
- **Data Split:** 80% Train / 20% Test (stratified)

> Turkish was added as a bonus language beyond the original assignment requirements.

---

## Feature Extraction

Each `.wav` file is represented by a **52-dimensional feature vector** derived from 13 MFCCs:

| Statistic | Dimensions |
|-----------|------------|
| Mean      | 13         |
| Std Dev   | 13         |
| Max       | 13         |
| Min       | 13         |

A bias term `x0 = 1` is manually prepended to each feature vector as required.

---

## Model: From Scratch

Implemented in pure NumPy with the following components:

- **Softmax** — numerically stable (`z - max(z)`)
- **Cross-Entropy Loss** — with L2 regularization
- **Batch Gradient Descent** — with bias term excluded from regularization
- **Feature Noise** — small Gaussian noise added during training for regularization
- **Train/Validation tracking** — loss and accuracy logged per epoch

---

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|-----------------|--------------|----------|
| From Scratch | ~80.77% | — | — | — |
| Scikit-learn | — | — | — | — |

---

## Hyperparameter Tuning

Grid search over:
- Learning rates: `[0.05, 0.1, 0.2]`
- L2 regularization (`lambda`): `[0.001, 0.01, 0.1]`

Best result achieved with `LR=0.08`, `Lambda=0.02`, `epochs=3000`.

---

## Project Structure

```
├── ML_Assignment_3.ipynb       # Full implementation (Colab notebook)
├── transcript.pdf              # Written transcripts of all recorded answers
└── README.md
```

---

## How to Run

1. Open `ML_Assignment_3.ipynb` in Google Colab
2. Mount your Google Drive and update the folder paths to point to your `.wav` recordings
3. Run all cells in order

```python
# Install dependencies
!pip install librosa soundfile
```

---

## Tech Stack

- Python
- NumPy
- librosa
- scikit-learn
- matplotlib / seaborn

---

## Course

**AI-372 — Machine Learning**  
Department of Artificial Intelligence, UMT  
Fall Semester 2025