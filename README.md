
# Emotion Detection Fall 2024 - Top Rank Kaggle Submission 🏆

![Leaderboard Screenshot](./assets/leaderboard_top1.png)

> **Kaggle Competition:** [Emotion Detection Fall 2024](https://www.kaggle.com/competitions/emotion-detection-fall-2024/overview)  
> **Author:** Anmol Dwivedi (Rank 1/60+ submissions)

---

## 📃 Problem Statement
The task is to build a **multi-label emotion classification model** that detects 11 possible emotions from a given tweet. A tweet can contain **multiple emotions** or none. The models were trained and evaluated using `train.csv` and tested on `test.csv` with the same schema (without label columns).

### Emotion Labels:
- `anger`, `anticipation`, `disgust`, `fear`, `joy`, `love`, `optimism`, `pessimism`, `sadness`, `surprise`, `trust`

---

## 🌐 Dataset Format
```csv
ID,Tweet,anger,anticipation,...,trust
2017-2144,"Worry is a down payment on a problem...",0,1,...,1
```
- **Input**: Free-form tweet text
- **Output**: Binary vector (length 11) indicating presence of each emotion

---

## 🔄 Workflow Overview

### 1. Data Preparation
- Combine binary label columns into a list of active emotions per tweet
- Apply `MultiLabelBinarizer` for one-hot encoding
- Convert `pandas` DataFrame to Hugging Face `Dataset`

### 2. Tokenization
Each notebook uses a different tokenizer:
- `Model_1`: `google/gemma-2-2b`
- `Model_2`: `meta-llama/Llama-3.2-1B`
- `Model_3`: `intfloat/e5-mistral-7b-instruct`

### 3. Training Setup
- 4-bit quantization with `BitsAndBytesConfig`
- LoRA configuration using `peft` for parameter-efficient tuning
- BCEWithLogitsLoss with `pos_weights` to handle class imbalance
- Gradient checkpointing for memory efficiency

### 4. Evaluation
- Custom metric computation using Hugging Face `evaluate`
- Optimization of decision thresholds for each label based on validation set

### 5. Inference
- Apply model to unseen test tweets
- Predict final labels using optimized thresholds
- Save predictions to `submission_model_X.csv` format

---

## 🎯 Models Used & Performance Comparison

### ✅ Evaluation Metrics (with Optimal Thresholds)
| Metric               | Model 1: Gemma-2B | Model 2: LLaMA-1B | Model 3: E5-Mistral-7B |
|----------------------|------------------|-------------------|------------------------|
| `f1_micro`           | 0.6909           | 0.6742            | ⭐ 0.7068            |
| `f1_macro`           | 0.6121           | 0.6102            | ⭐ 0.6366            |
| `accuracy_label`     | 0.8549           | 0.8417            | ⭐ 0.8630            |
| `accuracy_all`       | 0.1837           | 0.1759            | ⭐ 0.2199            |
| Inference Speed      | 🔢 Slow     | ⚡ Fast         | ✅ Moderate          |

### 🥇 Model Quick Comparison
| Model | Pros | Cons |
|-------|------|------|
| **Gemma-2B** | High performance on macro F1 and lowest loss. Stable across emotions. | Slower and heavier model. Needs more VRAM |
| **LLaMA-1B** | Lightweight, fastest training/inference. Great for limited GPUs | Slightly lower F1 and accuracy |
| **E5-Mistral-7B** | Best performance across all metrics. Great generalization. | Largest model, slower than LLaMA, needs more compute |

### 🌀 Summary:
- **Gemma** = Strong all-rounder, good F1 & loss
- **LLaMA** = Super fast, compact, ideal for Colab
- **E5-Mistral** = Best results, robust across all labels

---

## 📆 Notebooks Included
- `HW7_Model1_Gemma.ipynb` → Full workflow with `google/gemma-2-2b`
- `HW7_Model2_LLaMA.ipynb` → Lightweight inference with `meta-llama/Llama-3.2-1B`
- `HW7_Model3_E5Mistral.ipynb` → Embedding-based classifier using `e5-mistral-7b-instruct`

---

## 🎓 Leaderboard Achievement
Achieved **Rank 1** in the [Emotion Detection Fall 2024 Kaggle competition](https://www.kaggle.com/competitions/emotion-detection-fall-2024/overview) with a public score of **0.62227** ⭐

![Kaggle Leaderboard](./assets/kaggle_leaderboard_1stplace.png)

---

## 🚀 Future Plans
- Deploy best-performing model with Streamlit
- Add support for explainability using SHAP / LIME
- Test cross-dataset generalization (Twitter2020, EmotionX)

---

## 📅 Credits
- Built using Hugging Face Transformers, Datasets, Evaluate
- Optimized with LoRA & bitsandbytes quantization
- Competition hosted by UTD | Course: NLP - Fall 2024

---

## 📖 Citation
If you use this codebase, please cite the work or link back to this repo.

---

Feel free to ⭐ star the repo if this helped you!
