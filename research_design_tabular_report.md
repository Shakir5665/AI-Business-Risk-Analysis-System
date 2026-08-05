# Section 04 — Research Design & Tabular Workflow Report
## AI-Powered Business Risk Analysis and Recommendation System

---
 
### Overview of Research Component Architecture

The research component of the project focuses on formulating sentiment classification and aspect category identification from customer reviews as an end-to-end multi-task multi-label deep learning research model. 

The research model workflow spans raw data ingestion, preprocessing, neural encoding via XLM-RoBERTa bottleneck adapters, and prediction generation:

```
[ Raw Reviews ] ──> [ Preprocessing ] ──> [ XLM-R Backbone ] ──> [ Dual Adapters ] ──> [ Sentiment & Aspect Predictions ]
```

> **System Boundary Note**: The predicted sentiment labels and detected aspects produced by this research model become the direct input to the downstream **Business Risk Analysis module**, which is implemented and evaluated in **Implementation II**.

This document details the complete research lifecycle across seven sub-sections (4.1–4.7). Sub-sections 4.1 through 4.5 are presented in a standardized **Field-Table Format** containing six mandatory parameters:
1. **Name of the workflow unit**
2. **Objective**
3. **Input**
4. **Process in Brief**
5. **Output**
6. **Business rules / constraints / Assumptions**

---

### 4.1 Data/Input Use

The research component relies on a specialized dataset composed of multilingual customer reviews collected from e-commerce platforms (e.g. Daraz) and social commerce channels (Facebook, Instagram, WhatsApp) operating in Sri Lanka and regional South Asian markets. The dataset incorporates code-switched text combining Sinhala script, Singlish (Sinhala written in Latin characters), English, and informal digital commercial slang.

#### Standardized Workflow Unit Specification: Data/Input Use

| Parameter | Field Detail & Technical Description |
| :--- | :--- |
| **Name of the workflow unit** | `Multilingual Customer Review Dataset Ingestion & Validation Unit` |
| **Objective** | To ingest, validate, and structure raw, unstructured customer review comments into a standardized format for deep learning model ingestion. |
| **Input** | Raw JSONL records located at [data/raw/dataset.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/raw/dataset.jsonl) containing attributes:<br>• `review_id` (String): Unique identifier (e.g. `"rev_008"`) <br>• `text` (String): Unstructured text comment<br>• `sentiment` (String): Ground truth polarity (`"negative"`, `"neutral"`, `"positive"`)<br>• `aspects` (Array of Strings): Operational aspect ground truth (`["quality"]`, `["delivery", "trust"]`) |
| **Process in Brief** | 1. Ingest JSONL records into memory via pandas DataFrame loader ([dataset_loader.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/dataset_loader.py)).<br>2. Execute schema validation checks using `DatasetValidator` ([validator.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/validator.py)) to ensure non-empty text strings and valid sentiment/aspect labels.<br>3. Calculate corpus statistics (token length distribution, vocabulary size, label frequency). |
| **Output** | Validated PyTorch-ready DataFrame with 12,540 total sample records ready for split processing ([dataset.jsonl](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/data/final/dataset.jsonl)). |
| **Business rules / constraints / Assumptions** | • **Constraint**: All reviews must contain non-null `text` attributes.<br>• **Business Rule**: Sentiment labels must strictly belong to `{"negative", "neutral", "positive"}` ([model_labels.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_labels.py#L18-L26)).<br>• **Assumption**: Aspect labels are multi-label subset of `{"quality", "trust", "delivery"}` ([model_labels.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_labels.py#L32-L40)). |

---

### 4.2 Data Preprocessing

Raw customer text contains noise, non-standard spellings, repetitive letters, and regional code-switching. The preprocessing workflow standardizes text before passing it to the neural tokenizer.

#### Standardized Workflow Unit Specification: Data Preprocessing

| Parameter | Field Detail & Technical Description |
| :--- | :--- |
| **Name of the workflow unit** | `Multilingual Text Preprocessing & Label Encoding Unit` |
| **Objective** | To clean raw text, normalize Singlish/Sinhala slang, remove orthographic noise, encode labels into numerical tensors, and generate subword token IDs. |
| **Input** | Raw review text string $S$ and ground-truth label targets (`sentiment` string, `aspects` string list). |
| **Process in Brief** | 1. **Text Cleaning** ([cleaner.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/cleaner.py)): Unicode NFC normalization, lowercasing, decoding HTML entities, regex removal of URLs, emails, user handles, backslashes, brackets, and quotes.<br>2. **Character Repeat Normalization** ([repeat_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/repeat_normalizer.py)): Truncates character sequences repeated $\ge 3$ times down to 2 characters via regex `(.)\1{2,}`.<br>3. **Sri Lankan Slang Normalization** ([srilankan_normalizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/preprocessing/srilankan_normalizer.py)): Replaces Singlish, Sinhala, and commercial slang terms with standard words using [slang_dictionary.json](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/resources/slang_dictionary.json) (e.g. `dlvry` $\rightarrow$ `delivery`, `boru` $\rightarrow$ `fake`).<br>4. **Subword Tokenization** ([tokenizer.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/tokenization/tokenizer.py)): Encodes normalized text with `FacebookAI/xlm-roberta-base` SentencePiece tokenizer, padding/truncating to sequence length 512.<br>5. **Target Encoding**: `SentimentEncoder` ([sentiment_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/sentiment_encoder.py)) maps sentiment to single integer $\in \{0, 1, 2\}$; `AspectEncoder` ([aspect_encoder.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/encoding/aspect_encoder.py)) converts aspect lists to 3-dim multi-hot binary vectors. |
| **Output** | Dictionary containing PyTorch Tensors:<br>• `input_ids`: Tensor of shape `(batch_size, 512)`<br>• `attention_mask`: Tensor of shape `(batch_size, 512)`<br>• `sentiment`: Long Tensor of shape `(batch_size,)`<br>• `aspects`: Float Tensor of shape `(batch_size, 3)` |
| **Business rules / constraints / Assumptions** | • **Constraint**: Sequence length is strictly capped at `MAX_SEQUENCE_LENGTH = 512` ([model_config.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/configs/model_config.py#L19)).<br>• **Business Rule**: Unrecognized words outside dictionary fall back to original token subwords.<br>• **Assumption**: Unicode NFC composition preserves all valid Sinhala diacritics. |

---

### 4.3 Model / Algorithm

The core research component employs a **Multi-Task Bottleneck Adapter Model** built on top of a frozen pre-trained XLM-RoBERTa encoder backbone.

#### Standardized Workflow Unit Specification: Model / Algorithm

| Parameter | Field Detail & Technical Description |
| :--- | :--- |
| **Name of the workflow unit** | `Multi-Task Adapter Neural Inference Engine` |
| **Objective** | To generate joint probability distributions for sentiment classification and aspect category identification from contextual sequence embeddings. |
| **Input** | `input_ids` Tensor $\mathbf{X}_{\text{ids}} \in \mathbb{R}^{B \times 512}$ and `attention_mask` Tensor $\mathbf{X}_{\text{mask}} \in \mathbb{R}^{B \times 512}$. |
| **Process in Brief** | 1. **Backbone Forward Pass** ([backbone.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/backbone.py)): Extracts sequence representation from index 0 `[CLS]` token: $\mathbf{h}_{\text{CLS}} = \text{XLM-R}(\mathbf{X}_{\text{ids}}, \mathbf{X}_{\text{mask}})[:, 0, :] \in \mathbb{R}^{B \times 768}$. Backbone parameters are frozen (`FREEZE_BACKBONE = True`).<br>2. **Sentiment Branch Bottleneck Adapter** ([adapter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/adapter.py)): Projects 768 dimensions down to 128 bottleneck dimensions, applies ReLU and 0.1 dropout, projects back to 768, and adds residual connection: $\mathbf{f}_{\text{sent}} = \text{Adapter}_{\text{sent}}(\mathbf{h}_{\text{CLS}}) + \mathbf{h}_{\text{CLS}}$.<br>3. **Aspect Branch Bottleneck Adapter**: Computes task-specific feature vector: $\mathbf{f}_{\text{asp}} = \text{Adapter}_{\text{asp}}(\mathbf{h}_{\text{CLS}}) + \mathbf{h}_{\text{CLS}}$.<br>4. **Classification Heads** ([classification_heads.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/models/classification_heads.py)):<br>• Sentiment Head: $\mathbf{z}_{\text{sent}} = \text{Linear}_{256 \to 3}(\text{ReLU}(\text{Linear}_{768 \to 256}(\text{Dropout}(\mathbf{f}_{\text{sent}})))) \in \mathbb{R}^{B \times 3}$<br>• Aspect Head: $\mathbf{z}_{\text{asp}} = \text{Linear}_{256 \to 3}(\text{ReLU}(\text{Linear}_{768 \to 256}(\text{Dropout}(\mathbf{f}_{\text{asp}})))) \in \mathbb{R}^{B \times 3}$ |
| **Output** | Dictionary containing raw prediction logits:<br>• `sentiment_logits`: Float Tensor of shape `(batch_size, 3)`<br>• `aspect_logits`: Float Tensor of shape `(batch_size, 3)` |
| **Business rules / constraints / Assumptions** | • **Constraint**: Hidden dimension is fixed at $d=768$, bottleneck dimension $m=128$.<br>• **Business Rule**: Adapters train independently for each task while sharing frozen backbone weights.<br>• **Hyperparameter Summary**: $\text{LR}=5\times 10^{-5}$, $\text{Weight Decay}=0.01$, $\text{Batch Size}=64$, $\text{Epochs}=50$. |

---

### 4.4 Model Evaluation

The model evaluation workflow measures model performance using multi-task metrics and validates convergence using early stopping.

#### Standardized Workflow Unit Specification: Model Evaluation

| Parameter | Field Detail & Technical Description |
| :--- | :--- |
| **Name of the workflow unit** | `Multi-Task Loss Computation & Metrics Evaluator Unit` |
| **Objective** | To compute balanced loss metrics, evaluate classification performance on validation/test splits, and enforce early stopping. |
| **Input** | Predicted logits (`sentiment_logits`, `aspect_logits`) and ground-truth target tensors (`sentiment`, `aspects`). |
| **Process in Brief** | 1. **Class Weight Calculation** ([class_weights.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/class_weights.py)): Computes inverse frequency weights $w_c$ for CrossEntropy and positive weights $w_{\text{pos}, a}$ for BCE.<br>2. **Loss Computation** ([losses.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/losses.py)):<br>• $L_{\text{sentiment}} = \text{CrossEntropyLoss}(\mathbf{z}_{\text{sent}}, y_{\text{sent}}, \text{weight}=w_c)$<br>• $L_{\text{aspect}} = \text{BCEWithLogitsLoss}(\mathbf{z}_{\text{asp}}, y_{\text{asp}}, \text{pos\_weight}=w_{\text{pos}, a})$<br>• $L_{\text{total}} = L_{\text{sentiment}} + L_{\text{aspect}}$<br>3. **Evaluation Metrics** ([evaluator.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/evaluation/evaluator.py)): Computes Sentiment Accuracy, Macro Precision/Recall/F1, and Aspect Micro/Macro F1.<br>4. **Early Stopping** ([early_stopping.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/early_stopping.py)): Monitors validation total loss with patience=5 epochs and min_delta=0.001. |
| **Output** | Quantitative metric dictionary containing Loss values, Sentiment Accuracy (91.45%), Sentiment Macro F1 (90.97%), Aspect Micro F1 (89.64%), and confusion matrices. |
| **Business rules / constraints / Assumptions** | • **Constraint**: Stratified 80/10/10 dataset split ([splitter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/splitter.py)) based on sentiment label.<br>• **Business Rule**: Best model checkpoint saved automatically when validation loss improves ([checkpoint.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/checkpoint.py)). |

---

### 4.5 Output Application

This section specifies the immediate output produced by the research component and defines how these predictions serve as inputs to downstream system modules.

```
[ Raw Reviews ] ──> [ Preprocessing ] ──> [ XLM-R Backbone ] ──> [ Sentiment Branch ] ──> [ Sentiment Prediction ]
                                                              │                                       │
                                                              └──> [ Aspect Branch ] ───> [ Aspect Prediction ]
                                                                                                      │
                                                                                                      ▼
                                                                                   [ Handoff to Business Risk Analysis ]
                                                                                         (Implementation II Module)
```

#### Standardized Workflow Unit Specification: Output Application

| Parameter | Field Detail & Technical Description |
| :--- | :--- |
| **Name of the workflow unit** | `Model Prediction Output & Downstream Module Interface Unit` |
| **Objective** | To transform raw model logits into calibrated probability distributions and discrete class labels for sentiment polarity and active business aspects, establishing a clean handoff interface to downstream business logic modules. |
| **Input** | Model prediction logits (`sentiment_logits` $\in \mathbb{R}^3$, `aspect_logits` $\in \mathbb{R}^3$) generated by the trained `BusinessRiskModel`. |
| **Process in Brief** | 1. **Sentiment Probability & Class Extraction**:<br>• Applies Softmax function: $P(\text{sentiment}_c) = \text{Softmax}(\mathbf{z}_{\text{sent}})_c$<br>• Extracts predicted class label: $\hat{y}_{\text{sent}} = \text{argmax}_c P(\text{sentiment}_c) \in \{\text{negative}, \text{neutral}, \text{positive}\}$, accompanied by confidence percentage.<br>2. **Aspect Probability & Active Label Detection**:<br>• Applies Sigmoid activation per aspect: $P(\text{aspect}_a) = \sigma(z_{\text{asp}, a})$<br>• Identifies active aspect categories using decision threshold $\tau = 0.5$: $\hat{y}_{\text{asp}, a} = \mathbb{I}(P(\text{aspect}_a) \ge 0.5) \in \{\text{quality}, \text{trust}, \text{delivery}\}$.<br>3. **Downstream Interface Handoff**:<br>• Packages predictions into a standardized model prediction payload. The predicted sentiment labels and detected aspects become the direct input to the **Business Risk Analysis module**, which is implemented and evaluated in **Implementation II**. |
| **Output** | Structured Prediction Object:<br>• `sentiment`: Discrete sentiment label (`"negative"`, `"neutral"`, `"positive"`) + confidence score<br>• `aspects`: List of detected active aspect strings (e.g., `["quality", "delivery"]`) + per-aspect probability scores |
| **Business rules / constraints / Assumptions** | • **System Boundary Rule**: The research component terminates at prediction generation (Sentiment Label & Aspect List). It does not directly calculate risk scores or generate operational recommendations within this component.<br>• **Handoff Rule**: **"The predicted sentiment labels and detected aspects become the input to the Business Risk Analysis module, which is implemented and evaluated in Implementation II."**<br>• **Constraint**: Sigmoid decision threshold for active aspect filtering is fixed at $\tau = 0.5$. |

---

### 4.6 Validation of Research Component

Empirical validation of the research component demonstrates consistent high performance across both sentiment classification and aspect category identification on the held-out test dataset (1,252 samples):

#### 1. Confusion Matrix Analysis (Sentiment Classification)
The confusion matrix for 3-class sentiment classification demonstrates high diagonal concentration:

```
                   Predicted Negative   Predicted Neutral   Predicted Positive
Actual Negative :        412                  18                   15
Actual Neutral  :         22                 378                   19
Actual Positive :         12                  21                  355
```
- **Negative Class Precision**: $412 / (412 + 22 + 12) = 92.38\%$
- **Neutral Class Precision**: $378 / (18 + 378 + 21) = 90.65\%$
- **Positive Class Precision**: $355 / (15 + 19 + 355) = 89.42\%$
- **Overall Model Accuracy**: **91.45%**

#### 2. Aspect Classification Performance Metrics
Multi-label aspect classification achieved robust performance across all operational categories:

| Aspect Category | Precision | Recall | F1-Score | Evaluation Basis |
| :--- | :---: | :---: | :---: | :--- |
| **Quality** | 92.15% | 90.27% | **91.20%** | Product defects, material issues |
| **Trust** | 88.40% | 87.21% | **87.80%** | Misleading specs, seller behavior |
| **Delivery** | 86.90% | 87.60% | **87.25%** | Transit delay, box damage |
| **Micro Average** | 89.85% | 89.43% | **89.64%** | Global sample evaluation |
| **Macro Average** | 89.15% | 88.36% | **88.75%** | Equal category weighting |

#### 3. Stratified Loss Convergence
During training, joint multi-task loss demonstrated smooth convergence without overfitting due to early stopping:
- Initial Training Loss (Epoch 1): Total = 1.4820 (Sentiment = 0.8120, Aspect = 0.6700)
- Optimal Validation Loss (Epoch 14): Total = 0.4826 (Sentiment = 0.2841, Aspect = 0.1985)
- Early stopping triggered at Epoch 19, confirming optimal model snapshot selection.

---

### 4.7 Data Design

The data design of the system is specifically structured to support **multi-task multi-label transformer learning** on informal regional text.

#### 1. Data Schema & Record Definition
Data records are formatted as JSON Lines (`.jsonl`) to enable memory-efficient streaming and atomic sample processing:

```json
{
  "review_id": "rev_12448",
  "text": "පැකින් හොදයි. ඉක්මනට එව්ව. එකම අවුල අද මට මේක ලැබුන. ඩිලිවරි free කරල මට ගන්න 2278/- ගියා.",
  "sentiment": "negative",
  "aspects": ["delivery", "trust"]
}
```

#### 2. Structural Alignment with Neural Architecture
- **Text Attribute**: Supports raw unicode strings containing Sinhala script (`U+0D80` to `U+0DFF`), Latin characters, numerals, punctuation, and emojis.
- **Sentiment Attribute**: Categorical target mapped via `SentimentEncoder` to 3-class index $\{0, 1, 2\}$ for standard Cross-Entropy Optimization.
- **Aspects Attribute**: Variable-length list of tags mapped via `AspectEncoder` to a fixed-length 3-dimensional multi-hot binary vector $[y_1, y_2, y_3] \in \{0, 1\}^3$. This format enables Binary Cross-Entropy with Logits loss ($L_{\text{BCE}}$) to evaluate each aspect independently as a non-mutually exclusive binary decision.

#### 3. Data Split & Balance Strategy
- **Stratified Partitioning**: Dataset split into Train (80%, 10,034 samples), Validation (10%, 1,254 samples), and Test (10%, 1,252 samples) using `train_test_split` with `stratify=dataframe["sentiment"]` and `random_state=42` ([splitter.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/dataset/splitter.py#L76-L86)).
- **Imbalance Mitigation**: Dynamic inverse frequency weighting in `ClassWeightCalculator` ([class_weights.py](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/src/training/class_weights.py#L38-L84)) ensures minority sentiment classes and lower-frequency aspect categories contribute proportionally to gradient updates during backward propagation.
