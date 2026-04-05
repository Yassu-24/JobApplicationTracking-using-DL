# 📬 AI-Powered Smart Job Application Tracker

> An end-to-end deep learning pipeline that automatically fetches, preprocesses, and classifies job-related emails from Gmail — built entirely from scratch using self-collected, self-labeled data.

**Course:** MSS-M-2: Case Study Machine Learning and Deep Learning (WS25/26) — Group A6  
**Team:** Sai Vamshi Kolakani · Chengala Rahul · Katkam Praneeth · Thanush

---

## 📌 Table of Contents

- [Motivation](#motivation)
- [How It Works — Overview](#how-it-works--overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [Step 1 — Gmail Fetch & Metadata Extraction](#step-1--gmail-fetch--metadata-extraction)
  - [Step 2 — Preprocessing](#step-2--preprocessing)
  - [Step 3 — Weak Labeling](#step-3--weak-labeling)
  - [Step 4 — Model Training (Phase 1)](#step-4--model-training-phase-1)
  - [Step 5 — Inference (Phase 2)](#step-5--inference-phase-2)
- [Models](#models)
- [Evaluation](#evaluation)
- [Project Timeline](#project-timeline)
- [Team](#team)

---

## Motivation

When actively applying for jobs, your inbox quickly fills with hundreds of emails — job alerts, promotional spam, acknowledgments, interview invites, and rejections — all mixed together with no structure.

**The problem:** There is no easy way to track where you stand across multiple applications without manually reading every email.

**This project solves that** by building an automated pipeline that:
- Connects to your Gmail via the Gmail API
- Pulls and cleans all incoming emails
- Classifies them first by *category* (job-related vs. noise), then by *intent* (interview / rejection / acknowledgment / next step)
- Outputs structured, labeled data so you always know the status of every application

The entire dataset was built from scratch — real Gmail data, no pre-existing labeled corpus — and labeled using weak supervision (keyword + sender-based rules), making this a fully self-contained, real-world NLP project.

---

## How It Works — Overview

The pipeline has two distinct phases:

| Phase | What happens |
|---|---|
| **Phase 1: Training** | Weak-labeled email data is used to train CNN and XLM-RoBERTa models for both classification stages. Trained models are saved to disk. |
| **Phase 2: Inference** | New Gmail emails are fetched live, preprocessed, and passed through the saved models to produce final intent labels. |

Within each phase, classification happens in **two stages with a filter in between**:

```
Incoming Email
     │
     ▼
[Stage 1]  Category Classifier  ──▶  Non-job / Spam / Promotional  ──▶  Ignore
     │
     │  Job-related only
     ▼
[Stage 1.5]  Authenticity / Leak Filter
     │
     ▼
[Stage 2]  Intent Classifier
     │
     ▼
Interview / Rejection / Acknowledgment / Next Step / Other
```

---

## System Architecture

### Full Pipeline Flow

```
Incoming Emails (Gmail)
         │
         ▼
   Gmail API Fetch
         │
         ▼
Extract Email Metadata
  • Subject  • Sender  • Date  • Body
         │
         ▼
      Preprocessing
  • Remove HTML tags
  • Normalize umlauts
  • Replace URLs & emails
  • Lowercase & strip
         │
         ▼
┌──────────────────────────────┐
│  Stage 1: Email Category     │
│  Classifier                  │
└────────────┬─────────────────┘
             │
     ┌───────┴──────────────┐
     │                      │
Non-job / Spam /        Job-related
Promotional                  │
     │                       ▼
     ▼          ┌────────────────────────┐
  [Ignore]      │  Stage 1.5:            │
                │  Leak / Authenticity   │
                │  Filter                │
                └──────────┬─────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Stage 2: Job Email   │
                │  Intent Classifier    │
                └──────────┬────────────┘
                            │
                            ▼
                Intent Types:
                • Interview
                • Rejection
                • Acknowledgment
                • Next Step
                            │
                            ▼
                Structured Output
                (Intent + Metadata)
```

### Two-Phase System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Phase 1: Dataset Training                │
│                                                          │
│   Stage-1 Dataset               Stage-2 Dataset          │
│   (Job vs Non-Job vs            (Job Intent              │
│    Spam vs Promotional)          Classification)         │
│         │                             │                  │
│         ▼                             ▼                  │
│   CNN + XLM-RoBERTa           CNN + XLM-RoBERTa         │
│   Training (Stage-1)          Training (Stage-2)         │
│         │                             │                  │
│         ▼                             ▼                  │
│   [Saved Stage-1 Model]       [Saved Stage-2 Model]      │
└──────────────┬───────────────────────┬───────────────────┘
               │     (load models)     │
┌──────────────▼───────────────────────▼───────────────────┐
│               Phase 2: Real Gmail Inference               │
│                                                          │
│   Gmail API Fetch Emails                                 │
│         │                                                │
│         ▼                                                │
│   Preprocessing + Tokenization                           │
│         │                                                │
│         ▼                                                │
│   Stage-1 Inference (Job vs Non-Job)                     │
│         │                                                │
│         ▼                                                │
│   Stage-1.5 Authenticity Filter                          │
│         │                                                │
│         ▼                                                │
│   Stage-2 Inference (Job Intent Classification)          │
│         │                                                │
│         ▼                                                │
│   Final Job Labels                                       │
└──────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | Real Gmail inbox (self-collected) |
| **Size** | 5,610 raw emails |
| **Fields** | Subject, Email Text, Date, Sender |
| **Labeling** | Weak labeling — keyword + sender-based rules (no manual annotation) |
| **Spot-check accuracy** | 85–90% |

This is a **fully self-built dataset** — no pre-labeled corpus was used. Labels are generated automatically using weak supervision rules, which are fast, scalable, and easy to update.

---

## Pipeline

### Step 1 — Gmail Fetch & Metadata Extraction

Emails are pulled from Gmail using the **Gmail API** and the following fields are extracted:

- `Subject`
- `Sender`
- `Date`
- `Body`

**Preprocessing example:**

| Field | Value |
|---|---|
| Subject (original) | `New job opportunities for you` |
| Email Text (raw) | `Picked for you! — Yasaswini, you deserve to love your job...` |
| preprocessed_text (final) | `new job opportunities for you picked for you yasaswini you deserve to love your job based on your search profile...` |

---

### Step 2 — Preprocessing

| Step | Operation |
|---|---|
| HTML removal | Strip all HTML tags from email body |
| URL replacement | Replace all URLs with placeholder token |
| Email removal | Remove email addresses from body text |
| Umlaut normalization | `ä → ae`, `ö → oe`, `ü → ue` |
| Lowercasing | Lowercase entire text |
| Text consolidation | Combine `Subject + Body → preprocessed_text` |
| Quality checks | Fill missing values; verify first 10–15 samples |

---

### Step 3 — Weak Labeling

No ground-truth labels exist, so pseudo-labels are generated using rule-based weak supervision across two stages.

**Stage 1 — Category Labels**

| Label | Keyword Rules | Sender-Based Rules |
|---|---|---|
| `job_related` | job, vacancy, career, position, opportunity, hiring, interview, application, recruiter | sender contains: recruiter, talent, hr, jobs, careers |
| `promotional` | unsubscribe, sale, promotion, deal, discount, coupon | sender contains: news, offers, newsletter |
| `spam` | "click here" OR "buy now" AND spam domains | sender contains: noreply, no-reply, mailer, postmaster |
| `job_agency_spam` | — | sender contains: indeed, stepstone, monster, glassdoor, linkedin, jooble |
| `non_job` | DEFAULT — no rules matched | — |

**Stage 2 — Intent Labels** *(applied only to `job_related` emails)*

| Label | Pattern / Keywords |
|---|---|
| `interview` | interview, interview scheduled, we would like to meet, schedule an interview, invite you |
| `rejection` | reject, unfortunately, not selected, we regret to inform |
| `acknowledgment` | thank you for applying, thanks for your application, acknowledge |
| `next_step` | next steps, please complete, follow up, task, assessment, assignment |
| `other` | DEFAULT — no patterns matched |

---

### Step 4 — Model Training (Phase 1)

Two independent models are trained for each classification stage:

**Stage 1 — Email Category Classification**

| Model | Role |
|---|---|
| CNN | Fast baseline using text embeddings |
| XLM-RoBERTa (base) | Transformer for higher accuracy across multilingual email content |

**Stage 2 — Job Intent Classification**

| Model | Role |
|---|---|
| CNN | Lightweight classifier for intent patterns |
| XLM-RoBERTa (base) | Captures subtle contextual cues (interview vs. rejection vs. next step) |

All trained models are **saved to disk** and loaded during Phase 2 inference.

---

### Step 5 — Inference (Phase 2)

New emails from Gmail go through the full pipeline using the saved models:

1. **Gmail API** fetches live emails
2. **Preprocessing + Tokenization** cleans and prepares text
3. **Stage-1 Inference** — classify as Job vs. Non-Job
4. **Stage-1.5 Authenticity Filter** — remove leakage / false positives
5. **Stage-2 Inference** — classify job emails by intent
6. **Final labels** output as structured data (Intent + Metadata)

---

## Models

| Stage | Model | Input | Output Labels |
|---|---|---|---|
| Stage 1 | CNN | preprocessed_text | job_related, promotional, spam, job_agency_spam, non_job |
| Stage 1 | XLM-RoBERTa | preprocessed_text | job_related, promotional, spam, job_agency_spam, non_job |
| Stage 2 | CNN | job_related emails only | interview, rejection, acknowledgment, next_step, other |
| Stage 2 | XLM-RoBERTa | job_related emails only | interview, rejection, acknowledgment, next_step, other |

**Optimization strategies:**
- Hyperparameter tuning (batch size, learning rate, epochs)
- Improved preprocessing (lemmatization, bigrams, stopword tuning)
- Label noise reduction via refined weak labeling rules
- Ensemble predictions (CNN + XLM-RoBERTa) for higher accuracy

---

## Evaluation

Each model is evaluated on:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Avoid false positives |
| Recall | Avoid missing genuine job emails |
| F1-score (macro) | Balanced performance across all classes |
| Confusion Matrix | Identify which intent types are confused |

**Comparison studies:**
- CNN vs. XLM-RoBERTa on Stage 1 (category classification)
- CNN vs. XLM-RoBERTa on Stage 2 (intent detection)

