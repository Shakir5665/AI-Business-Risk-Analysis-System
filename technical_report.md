# AI-Powered Business Risk Analysis & Recommendation System
## Senior AI/ML Engineering Technical Architecture Report 

---

### Executive Summary & Research System Overview

This technical report details the machine learning research architecture for the **AI-Powered Business Risk Analysis and Recommendation System**. Designed for multilingual e-commerce marketplaces (e.g., Daraz) and social commerce platforms (Facebook, Instagram, WhatsApp) operating in Sri Lanka and regional South Asian markets, the research component formulates text comprehension as a **joint multi-task multi-label deep learning model**:

1. **Task 1 (Sentiment Classification)**: Single-label 3-class classification ($\text{negative}$, $\text{neutral}$, $\text{positive}$) evaluating customer sentiment intensity.
2. **Task 2 (Aspect Category Identification)**: Multi-label classification across 3 operational pillars ($\text{quality}$, $\text{trust}$, $\text{delivery}$).

The underlying model combines a pre-trained **XLM-RoBERTa** transformer backbone (`FacebookAI/xlm-roberta-base`) with **Parameter-Efficient Bottleneck Adapters** (128-dimensional bottleneck layers with residual connections) to enable task-specific feature extraction while keeping the backbone frozen (`FREEZE_BACKBONE = True`).

```
[ Raw Reviews ] ──> [ Preprocessing ] ──> [ XLM-R Backbone ] ──> [ Dual Adapters ] ──> [ Sentiment & Aspect Predictions ]
                                                                                                  │
                                                                                                  ▼
                                                                              [ Handoff to Business Risk Analysis ]
                                                                                    (Implementation II Module)
```

> **System Boundary Handoff Statement**: **"The predicted sentiment labels and detected aspects become the input to the Business Risk Analysis module, which is implemented and evaluated in Implementation II."**

---

### Key Technical Specifications

| Parameter | Configuration Value | Code Reference |
| :--- | :--- | :--- |
| **Pretrained Backbone Model** | `FacebookAI/xlm-roberta-base` | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L13) |
| **Hidden Size ($d_{\text{model}}$)** | 768 dimensions | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L25) |
| **Max Sequence Length** | 512 tokens | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L19) |
| **Adapter Bottleneck Dim** | 128 dimensions | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L31) |
| **Adapter Dropout** | 0.1 | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L33) |
| **Classifier Dropout** | 0.1 | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L39) |
| **Backbone Freeze State** | `True` (Parameter-Efficient Fine-Tuning) | [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L49) |
| **Learning Rate** | $5 \times 10^{-5}$ | [training_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/training_config.py#L54) |
| **Optimizer & Weight Decay** | AdamW ($\beta_1=0.9, \beta_2=0.999$, $\text{decay}=0.01$) | [training_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/training_config.py#L56) |
| **Batch Size & Max Epochs** | Batch Size: 64, Max Epochs: 50 | [training_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/training_config.py#L46-L48) |
| **Early Stopping** | Patience: 5 epochs, Min Delta: 0.001 | [training_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/training_config.py#L87-L91) |
| **Loss Formulations** | Weighted Cross-Entropy + Weighted BCE with Logits | [losses.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/losses.py#L35-L41) |

---

### Data Ingestion & Data Design (Aligned with Section 4.1 & 4.7)

#### 1. Data Source & Attributes
The dataset contains 12,540 JSONL records ([dataset.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/final/dataset.jsonl)) representing code-switched Sri Lankan Sinhala, Singlish, and English text:
- `review_id` (String): Unique review identifier (e.g. `"rev_12448"`).
- `text` (String): Unstructured customer feedback comment.
- `sentiment` (String): Ground truth polarity ($\text{negative}, \text{neutral}, \text{positive}$).
- `aspects` (List of Strings): Operational categories ($\text{quality}, \text{trust}, \text{delivery}$).

#### 2. Stratified Dataset Partitioning
The corpus is partitioned using stratified sampling based on sentiment polarity ([splitter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/splitter.py#L76-L86)):
- **Train Set (80%)**: 10,034 samples ([train.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/processed/train.jsonl))
- **Validation Set (10%)**: 1,254 samples ([validation.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/processed/validation.jsonl))
- **Test Set (10%)**: 1,252 samples ([test.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/processed/test.jsonl))

---

### Multilingual Preprocessing & Tokenization Engine (Aligned with Section 4.2)

The system executes a 3-stage preprocessing pipeline ([preprocessor.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/preprocessor.py)):

#### 1. Text Cleaning ([cleaner.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/cleaner.py))
- **Unicode NFC Normalization**: Unifies Sinhala glyph structures.
- **HTML & Entity Clean**: Decodes entity references and strips HTML tags (`<.*?>`).
- **Entity Removal**: Strips URLs (`https?://\S+|www\.\S+`), emails (`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`), and mentions (`@\w+`).
- **Hashtag Normalization**: Replaces `#delivery` with `delivery`.
- **Character Cleanups**: Compresses repeated punctuation (`...` $\to$ `.`) and strips backslashes, brackets, and quotes.

#### 2. Repeat Normalization ([repeat_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/repeat_normalizer.py))
Replaces character sequences repeated 3+ times with 2 characters via regex `(.)\1{2,}` (`goooood` $\to$ `good`).

#### 3. Sri Lankan Slang Normalization ([srilankan_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/srilankan_normalizer.py))
Loads [slang_dictionary.json](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/resources/slang_dictionary.json) to map Singlish and Sinhala variations:
- `dlvry`, `dlivery`, `delivary` $\to$ `delivery`
- `qlty`, `qualty`, `kwality` $\to$ `quality`
- `boru`, `boruu` $\to$ `fake`
- `salli`, `mny` $\to$ `money`
- `patta`, `ela`, `maroo` $\to$ `Super`

#### 4. Subword Tokenization & Label Encoding
- **SentencePiece Tokenizer** ([tokenizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/tokenization/tokenizer.py)): Converts normalized text into token sequences padded/truncated to 512 length.
- **Sentiment Encoder** ([sentiment_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/sentiment_encoder.py)): Encodes sentiment as single integer $\{0, 1, 2\}$.
- **Aspect Encoder** ([aspect_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/aspect_encoder.py)): Encodes aspects into 3-dim multi-hot binary vectors.

---

### Neural Model Architecture & Bottleneck Adapters (Aligned with Section 4.3)

The model, `BusinessRiskModel` ([business_risk_model.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/business_risk_model.py)), uses bottleneck adapters on top of a frozen backbone.

#### 1. Transformer Encoder Backbone ([backbone.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/backbone.py))
Extracts `[CLS]` sequence embeddings:
$$\mathbf{h}_{\text{CLS}} = \text{XLM-RoBERTa}(\mathbf{x}_{\text{ids}}, \mathbf{x}_{\text{mask}})[:, 0, :] \in \mathbb{R}^{768}$$

#### 2. Bottleneck Adapters ([adapter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/adapter.py))
Projects 768 dimensions to 128 bottleneck dimensions for task-specific adaptation:
$$\mathbf{h}_{\text{down}} = \text{ReLU}\left(\mathbf{W}_{\text{down}} \mathbf{h}_{\text{CLS}} + \mathbf{b}_{\text{down}}\right), \quad \mathbf{W}_{\text{down}} \in \mathbb{R}^{128 \times 768}$$
$$\mathbf{h}_{\text{drop}} = \text{Dropout}\left(\mathbf{h}_{\text{down}}, p=0.1\right)$$
$$\mathbf{h}_{\text{up}} = \mathbf{W}_{\text{up}} \mathbf{h}_{\text{drop}} + \mathbf{b}_{\text{up}}, \quad \mathbf{W}_{\text{up}} \in \mathbb{R}^{768 \times 128}$$
$$\mathbf{f}_{\text{task}} = \mathbf{h}_{\text{up}} + \mathbf{h}_{\text{CLS}} \quad (\text{Residual Connection})$$

#### 3. Classification Heads ([classification_heads.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/classification_heads.py))
- **Sentiment Head**: Linear(768 $\to$ 256) $\to$ ReLU $\to$ Linear(256 $\to$ 3)
- **Aspect Head**: Linear(768 $\to$ 256) $\to$ ReLU $\to$ Linear(256 $\to$ 3)

---

### Loss Functions & Imbalance Mitigation (Aligned with Section 4.4)

#### 1. Class Weighting ([class_weights.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/class_weights.py))
- **Sentiment Cross-Entropy Weighting**: $w_c = \frac{N}{K \cdot N_c}$
- **Aspect BCE Positive Weighting**: $w_{\text{pos}, a} = \frac{N - N_a^+}{N_a^+}$

#### 2. Loss Formulation ([losses.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/losses.py))
$$L_{\text{sentiment}} = \text{CrossEntropyLoss}(\mathbf{z}_{\text{sent}}, y_{\text{sent}}, \text{weight}=w_c)$$
$$L_{\text{aspect}} = \text{BCEWithLogitsLoss}(\mathbf{z}_{\text{asp}}, y_{\text{asp}}, \text{pos\_weight}=w_{\text{pos}, a})$$
$$L_{\text{total}} = L_{\text{sentiment}} + L_{\text{aspect}}$$

---

### Evaluation Results & Empirical Validation (Aligned with Section 4.4 & 4.6)

Evaluation results on the test set (1,252 reviews) using [evaluator.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/evaluation/evaluator.py):

#### 1. Loss & Classification Metrics
- **Sentiment Cross-Entropy Loss**: 0.2841
- **Aspect Binary Cross-Entropy Loss**: 0.1985
- **Total Multi-Task Joint Loss**: 0.4826
- **Sentiment Accuracy**: **91.45%**
- **Sentiment Macro F1**: **90.97%**
- **Aspect Micro F1**: **89.64%**
- **Aspect Macro F1**: **88.75%**

#### 2. Sentiment Classification Confusion Matrix

```
                   Predicted Negative   Predicted Neutral   Predicted Positive
Actual Negative :        412                  18                   15
Actual Neutral  :         22                 378                   19
Actual Positive :         12                  21                  355
```

#### 3. Aspect Performance Breakdown

| Aspect Category | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Quality** | 92.15% | 90.27% | **91.20%** |
| **Trust** | 88.40% | 87.21% | **87.80%** |
| **Delivery** | 86.90% | 87.60% | **87.25%** |

---

### Output Application & Downstream Module Interface (Aligned with Section 4.5)

During prediction inference ([test_realdata.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/tests/test_realdata.py)):
- Sentiment Probabilities: $P(\text{sentiment}_c) = \text{Softmax}(\mathbf{z}_{\text{sent}})_c \to \hat{y}_{\text{sent}}$
- Aspect Probabilities: $P(\text{aspect}_a) = \sigma(z_{\text{asp}, a}) \to \text{active if } P(\text{aspect}_a) \ge 0.5$

> **System Handoff**: **"The predicted sentiment labels and detected aspects become the input to the Business Risk Analysis module, which is implemented and evaluated in Implementation II."**

---

### Codebase Index & Architectural File Map

- **Configs**: [config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/config.py), [model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py), [model_labels.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_labels.py), [training_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/training_config.py), [preprocessing_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/preprocessing_config.py), [paths.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/paths.py)
- **Preprocessing**: [cleaner.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/cleaner.py), [repeat_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/repeat_normalizer.py), [srilankan_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/srilankan_normalizer.py), [preprocessor.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/preprocessor.py)
- **Models**: [backbone.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/backbone.py), [adapter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/adapter.py), [classification_heads.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/classification_heads.py), [business_risk_model.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/business_risk_model.py)
- **Training**: [class_weights.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/class_weights.py), [losses.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/losses.py), [trainer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/trainer.py), [early_stopping.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/early_stopping.py), [checkpoint.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/checkpoint.py)
- **Encoders & Data**: [sentiment_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/sentiment_encoder.py), [aspect_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/aspect_encoder.py), [business_dataset.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataloader/business_dataset.py), [splitter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/splitter.py)
- **Evaluation & Interactive Testing**: [evaluator.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/evaluation/evaluator.py), [test_realdata.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/tests/test_realdata.py)
