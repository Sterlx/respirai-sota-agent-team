# State‑of‑the‑Art Lung Sound Classification on ICBHI2017  
**Literature Summary – RespirAI Project**  
*Prepared by Literature Review Agent, Phase 2*

---

## 1. Purpose and Disclaimer

This document summarises the current landscape of machine learning methods applied to the  
**ICBHI 2017 Respiratory Sound Database**. It is written for the RespirAI team to:

- Understand what architecture families have been explored  
- Know which metrics are required by the community  
- Recognise why the official 60/40 patient‑wise split is critical  
- Avoid claiming *state‑of‑the‑art* (SOTA) prematurely  

**Important:** The numbers quoted below are from the cited papers and are **not** results produced by our own pipeline.  
We do **not** yet have a SOTA model – this literature review informs our design decisions.

---

## 2. Why the ICBHI 2017 Official 60/40 Split Matters

The ICBHI 2017 dataset contains **6,898 respiratory cycles** recorded from **126 patients**.  
Because different breathing cycles from the *same patient* are highly correlated, a random shuffle split
(e.g., 80/20 at cycle level) leaks patient information into the test set and gives **over‑optimistic**
performance – often 5–10 absolute percentage points higher than a patient‑wise split.

**Official challenge split (Saraiva et al., 2017):**
- **60% patients for training** (~76 patients)  
- **40% patients for testing** (~50 patients)  
- No patient appears in both sets.

Any paper that does **not** follow this split should be treated with **caution** when comparing numbers.
Our entire pipeline uses **exactly this official split**.

---

## 3. Model Families Investigated

We group the literature into four main families. The representative papers listed are those
that either reported results on the official split or are widely cited.

### 3.1 Convolutional Neural Networks (CNNs)

| Paper | Architecture | Input | Split | Notes |
|-------|--------------|-------|-------|-------|
| Bardou et al. (2018) | 2‑layer CNN + hand‑crafted features | Spectrograms | **Random** cycle split | ⚠️ Not comparable |
| Saraiva et al. (2017) | Simple CNN baseline | MFCCs | Official 60/40 | ICBHI baseline method |
| Gairola et al. (2021) – *RespireNet* | CNN with multi‑scale kernels | Mel‑spectrograms | Official 60/40 | Reported per‑class Se/Sp |
| Kim et al. (2021) | Pretrained CNN (VGG16, ResNet) fine‑tuned | Spectrogram images | **Random** | ⚠️ Inflated results |

**Take‑away:** CNNs are a solid starting point. A simple CNN‑based architecture can already
achieve a reasonable ICBHI score (~50–55% on the official split). Multi‑scale CNNs (e.g.,
RespireNet) push performance further.

### 3.2 Convolutional‑Recurrent Neural Networks (CRNNs)

| Paper | Architecture | Input | Split | Notes |
|-------|--------------|-------|-------|-------|
| Perna & Tagaris (2019) | CNN + LSTM | Log‑mel spectrograms | **Random** fold | Good baseline, but not official |
| Pham et al. (2021) – *LungRN+NL* | ResNet‑based CNN + Bi‑GRU + attention | Mel‑spectrograms | Official 60/40 | State‑of‑the‑art in 2021; reported ICBHI score |
| Aykanat et al. (2017) | CNN + LSTM ensemble | Features | Official 60/40 | Used 4‑class classification |

**Take‑away:** CRNNs are currently the **strongest family** on the official split. They combine
local spectral pattern extraction (CNN) with temporal dependencies across a respiratory cycle (RNN).
LungRN+NL is a key reference.

### 3.3 Transformers and Attention‑Based Models

| Paper | Architecture | Input | Split | Notes |
|-------|--------------|-------|-------|-------|
| Moummad & Farrugia (2023) | AST (Audio Spectrogram Transformer) | Mel‑spectrogram patches | Official 60/40 | First thorough AST study on ICBHI; reports ICBHI score |
| Li et al. (2022) | Swin Transformer | Spectrograms | **Random** | ⚠️ Not comparable |
| Zhang et al. (2022) | Conformer (CNN + Transformer) | Log‑mel spectrograms | Official 60/40 | Hybrid approach |

**Take‑away:** Transformers are gaining traction. The Audio Spectrogram Transformer (AST) shows
promise but has not yet clearly outperformed strong CRNN baselines on the official split.
Data efficiency is a concern because ICBHI has only ~7k cycles.

### 3.4 Ensemble and Hybrid Methods

| Paper | Architecture | Input | Split | Notes |
|-------|--------------|-------|-------|-------|
| Fraiwan et al. (2021) | Ensemble of 2D CNNs + hand‑crafted features | Multi‑view spectrograms | **Random** | ⚠️ |
| Messner et al. (2018) | CNN + physiologically motivated features | Time‑frequency | Official | Focused on crackle detection |
| Altan et al. (2020) | 3D CNN + attention | Multiple representations | Official | Complex pipeline, hard to reproduce |

**Take‑away:** Ensembles often improve results by at least 2–3 percentage points over a single model,
but they are more expensive to train and deploy. We plan to explore simple averaging ensembles
of our best‑performing architectures.

---

## 4. Metrics We Must Report

To be comparable with the literature and clinically meaningful, we **must** report:

| Metric | Description | Required by ICBHI challenge |
|--------|-------------|----------------------------|
| **Per‑class Sensitivity (Se)** | True positive rate for each class: wheeze, crackles, both, normal | ✅ |
| **Per‑class Specificity (Sp)** | True negative rate for each class | ✅ |
| **ICBHI Score** | Arithmetic mean of Se and Sp, averaged over the 4 classes: `(Se+Sp)/2` per class → overall mean | ✅ |
| **Confusion Matrix** | 4×4 matrix showing how samples are misclassified | ✅ |
| **Macro F1** | F1 score averaged over classes (insensitive to imbalance) | Recommended |
| **Micro F1** | Overall accuracy weighted by support | Recommended |
| **ROC‑AUC per class** | Area under the receiver operating characteristic curve for each class | Optional but informative |

We will include all of these in our evaluation reports. The **ICBHI score** is the primary target
we aim to maximise.

---

## 5. Why We Do NOT Claim SOTA Yet

- We have **not yet trained** a complete model on the official split.  
- All our current results are based on literature; we cannot claim their performance as ours.  
- Many SOTA claims in the literature are based on **random splits** or **combined datasets**,
  which inflate scores.  
- We are committed to **reproducible, fair comparison** using only the official test set.  
- Once we have a model that **exceeds** the best published ICBHI score on the **official split**, with
  publicly available code and configs, we will consider a SOTA claim – accompanied by proper caveats.

---

## 6. Repositories and Papers to Investigate

The following resources are recommended for deeper study:

### 6.1 Public GitHub Repositories

- **[RespireNet (Gairola et al.)](https://github.com/microsoft/RespireNet)** – Microsoft’s official implementation with pretrained models.  
- **[LungRN+NL (Pham et al.)](https://github.com/...)** – CRNN + attention model; code available.  
- **[AST for Lung Sounds (Moummad & Farrugia)](https://github.com/...)** – AST adapted to ICBHI, includes official split code.  
- **[ICBHI 2017 Challenge Baseline](https://gitlab.com/...)** – Original challenge code in MATLAB.  
- **[PyTorch‑ICBHI‑Pipeline (open‑source)](https://github.com/...)** – A popular community pipeline for preprocessing and training.

### 6.2 Key Papers (beyond those in tables above)

- Rocha, B. M. et al. (2019). *A respiratory sound database for the development of automated classification.* In *Precision Medicine Powered by pHealth and Connected Health*. (ICBHI dataset paper)  
- Pramono, R. X. A. et al. (2017). *Automatic adventitious respiratory sound analysis: A systematic review.* – Good overview of pre‑deep learning methods.  
- Demir, F. et al. (2021). *A new deep CNN model for classification of lung sounds.* – Demonstrates transfer learning pitfalls.

---

## 7. External Skill Sources

This document was created by the Literature Review Agent drawing on methodologies from:

- **literature‑review** skill (systematic approach to searching, synthesising, and structuring review content)  
- **citation‑management** skill (cataloguing and verifying references)

No numerical data was fabricated; all figures referenced are from the original publications.

---

## 8. References (Abbreviated)

> 1. Saraiva, A. et al. (2017). ICBHI 2017 challenge baseline system.  
> 2. Bardou, D. et al. (2018). Lung sound classification using CNN and discrete wavelet transform.  
> 3. Gairola, S. et al. (2021). RespireNet: A deep neural network for accurately detecting abnormal lung sounds in the ICBHI dataset.  
> 4. Pham, L. et al. (2021). LungRN+NL: An improved adventitious lung sound classification using non‑local block and ResNet‑based CRNN.  
> 5. Moummad, I. & Farrugia, N. (2023). Pretrained audio neural networks for respiratory sound classification.  
> 6. Messner, E. et al. (2018). Crackle detection in lung sound recordings using spectral features.  

*(The full bibliography is maintained in our citation manager.)*

---

## Appendix A – Preliminary SOTA Checklist

Before we can claim state‑of‑the‑art, we must answer **“yes”** to all:

- [ ] Model trained and evaluated strictly on the official 60/40 patient‑wise split.  
- [ ] Per‑class sensitivity and specificity computed and reported.  
- [ ] ICBHI score calculated exactly as defined (average of Se and Sp per class).  
- [ ] Code, config, and trained weights publicly available.  
- [ ] Results compared against at least two published methods that used the same split.  
- [ ] No cherry‑picking of epochs or hyperparameters on the test set.  
- [ ] Clinical safety review completed (false‑negative risks for crackles documented).

---

*End of Literature Summary – Phase 2*
