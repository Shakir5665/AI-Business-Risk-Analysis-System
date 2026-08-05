# AI-Powered Business Risk Analysis & Recommendation System
## Executive Non-Technical & System Vision Report

---

### Executive Overview 

In today's digital economy, customer reviews flow constantly across e-commerce platforms (such as Daraz) and social commerce channels (Instagram, Facebook, WhatsApp). For merchants and marketplace operators, these reviews contain essential early warnings regarding operational issues.

However, manually reading thousands of feedback comments is impossible, and conventional keyword software fails when handling complex, informal, code-switched regional communications—such as mixed Sinhala, Singlish (Sinhala written in Latin script), and local commercial slang.

The **AI Research Model** addresses this challenge by providing intelligent 24/7 text analysis. It automatically processes customer reviews, determines customer sentiment polarity (Positive, Neutral, Negative), and identifies specific operational concern categories (Product Quality, Seller Trust, or Delivery Logistics).

```
[ Raw Reviews ] ──> [ Smart Preprocessing ] ──> [ Multilingual XLM-R AI ] ──> [ Sentiment & Aspect Predictions ]
                                                                                               │
                                                                                               ▼
                                                                           [ Handoff to Business Risk Analysis ]
                                                                                 (Implementation II Module)
```

> **System Boundary Note**: **"The predicted sentiment labels and detected aspects become the input to the Business Risk Analysis module, which is implemented and evaluated in Implementation II."**

---

### Key Business Challenges Addressed by the Research Model

1. **Multilingual & Singlish Communication Barriers**: Customers frequently mix languages within a single comment (e.g., *"seller හොඳයි, packing ok, but delivery tikak late"*). Standard AI systems fail on such text; our research model is specifically optimized to comprehend Sri Lankan digital dialect, phonetically spelled Sinhala, and commercial slang.
2. **Identification of Specific Operational Concerns**: A generic "negative review" does not inform managers about what went wrong. The research model categorizes every comment into specific operational pillars:
   - **Quality Aspects**: Defective items, broken parts, poor material, packaging damage.
   - **Trust Aspects**: Misleading product descriptions, fake claims, seller non-responsiveness, refund friction.
   - **Delivery Logistics Aspects**: Late delivery, courier delays, transit packaging damage.
3. **Automated Handoff for Risk Mitigation**: By providing high-accuracy sentiment and aspect predictions, the model enables downstream business modules to immediately trigger operational interventions.

---

### How the AI Pipeline Works (In Plain English)

The research AI pipeline operates through 4 distinct stages:

#### Stage 1: Text Cleaning & Multilingual Normalization
Raw reviews often contain digital noise like elongated words (`gooddddd!`), URLs, user handles, or informal abbreviations (`dlvry`, `qlty`, `salli`). The system cleans this noise and normalizes slang using a Sri Lankan domain dictionary ([slang_dictionary.json](file:///c:/Users/shaki/Desktop/Sentiment%20Model%20-%20AI-Business-Risk-Analysis-System/resources/slang_dictionary.json)).

#### Stage 2: Multilingual Transformer Understanding
Powered by `FacebookAI/xlm-roberta-base`, the AI model analyzes contextual sentence structures across Sinhala, Singlish, and English simultaneously, grasping the exact intent of the customer.

#### Stage 3: Dual-Task AI Predictions
The model processes the contextual representation through specialized neural adapters to generate dual predictions in a single step:
- **Sentiment Polarity**: Classifies emotional tone into **Negative**, **Neutral**, or **Positive** (achieving 91.45% accuracy).
- **Aspect Categories**: Identifies active categories among **Quality**, **Trust**, and **Delivery** (achieving 89.64% micro F1 score).

#### Stage 4: Prediction Handoff to Implementation II
The predicted sentiment labels and active aspects are packaged into a standardized output payload. This payload is passed directly to the downstream **Business Risk Analysis module** (Implementation II), which computes Business Risk Scores and manages automated operational workflows.

---

### Downstream Operational Handoff Matrix (Implementation II Integration)

The table below illustrates how the research model's predictions serve as inputs to the downstream Implementation II module:

| Predicted Sentiment | Detected Aspect | Downstream Operational Context | Target Business Intervention (Implementation II) |
| :--- | :--- | :--- | :--- |
| **Negative** | **Quality** | Product defect / damaged item | Initiate supplier batch quality audit; issue customer replacement or refund. |
| **Negative** | **Trust** | Seller misrepresentation / dispute | Flag seller store for compliance verification; temporarily pause store payouts. |
| **Negative** | **Delivery** | Transit delay / courier issue | File SLA ticket with 3PL courier provider; route future orders via alternative courier. |
| **Negative** | **Quality + Trust** | Severe fraud / fake product suspicion | Unpublish listing immediately; initiate formal vendor compliance review. |
| **Neutral** | **Delivery** | Shipping status inquiry | Dispatch automated tracking status SMS/WhatsApp notification to buyer. |
| **Positive** | **All Aspects** | Customer satisfaction | Highlight review as verified testimonial on seller product page. |

---

### Strategic Business Value & Implementation Roadmap

Implementing this AI research model provides significant strategic advantages:

1. **High Predictive Accuracy**: Delivers 91.45% Sentiment Accuracy and 89.64% Aspect Micro F1 across multilingual, code-switched reviews.
2. **Scalable Architecture**: Parameter-efficient bottleneck adapters allow lightweight model deployment without heavy computation overhead.
3. **Seamless Enterprise Integration**: Clear API separation ensures predicted outputs transition smoothly into downstream risk scoring and workflow systems.

```
Phase 1: Integration (Weeks 1-2)
  ├── Connect social media APIs (Facebook, Instagram, WhatsApp) and e-commerce platforms
  └── Ingest raw reviews into the Multilingual AI Preprocessing Pipeline

Phase 2: Prediction Pipeline Rollout (Weeks 3-4)
  ├── Deploy XLM-R Bottleneck Adapter Model for real-time Sentiment & Aspect inference
  └── Handoff prediction outputs to Implementation II Business Risk Module

Phase 3: Automated Workflow Enforcement (Month 2+)
  ├── Automate trigger-based refunds, replacement dispatches, and SLA ticket generation
  └── Conduct monthly executive reviews of vendor quality performance
```
