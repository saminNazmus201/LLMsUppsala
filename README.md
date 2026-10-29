# Large Language Models – Uppsala University

Hands-on projects from the **Large Language Models course (1RT730)** at Uppsala University: building LLM applications with open-weight models and evaluating how trustworthy they are.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)
![Qwen](https://img.shields.io/badge/Models-Qwen2.5--VL%20%7C%20Qwen3-6E56CF)

---

## Projects

| # | Project | Focus | Model | Highlights | Files |
|---|---------|-------|-------|------------|-------|
| 1 | [Mushroom Friend – Multimodal Chatbot](#1-mushroom-friend--multimodal-chatbot) | Building an LLM app | Qwen2.5-VL-3B-Instruct | Image + text chat, structured JSON memory, streaming Gradio UI | [`chatbotmushroom.ipynb`](chatbotmushroom.ipynb) · [report](chatbot1.pdf) |
| 2 | [Evaluating and Improving LLM Trustworthiness](#2-evaluating-and-improving-llm-trustworthiness) | Evaluation and safety | Qwen3-4B | Ethics, bias and privacy tests on DecodingTrust; password leakage cut from **92% → 0%** | [`Trustworthiness_Evaluation/`](Trustworthiness_Evaluation/) |
| … | *More projects coming as the course continues* | | | | |

---

## 1. Mushroom Friend – Multimodal Chatbot

**Goal:** Build a chatbot that can talk about mushrooms and analyse mushroom photos, working with text only, an image only, or both together.

### How it works

```
User image ──► Image-analysis prompt ──► Qwen2.5-VL ──► JSON record
                                                         │
User question ─────────────────────────────────────────► ├──► Answer grounded
                                                         │    in the stored JSON
System prompt (mushrooms only, safety first) ───────────►┘
                                     │
                                     ▼
                         Streamed reply in Gradio UI
```

1. When an image is uploaded, the model extracts a structured **JSON record**: common name, genus, confidence, visible parts, colour and edibility.
2. Follow-up questions are answered **from that stored JSON**, so the bot keeps a consistent memory of the mushroom being discussed.
3. A **system prompt** restricts the conversation to mushrooms and fungi, and steers it towards safety when toxic species come up.
4. Answers are **streamed token by token** into a Gradio chat interface using `TextIteratorStreamer` and a background thread.

### Evaluation

The chatbot was tested on ten criteria, documented in the [report](chatbot1.pdf):

| Criterion | Finding |
|-----------|---------|
| Prediction consistency | Fully reproducible with greedy decoding (`do_sample=False`) |
| Topic control | Politely redirects off-topic questions back to mushrooms |
| Transcription | Correctly read and summarised a Swedish field-guide page |
| JSON output | Always valid and correctly filled |
| Dangerous knowledge | Warned against eating *Amanita muscaria* and explained the risks |
| Classification accuracy | **Not reliable**: the 3B model often misidentified species while sounding confident |

**Key insight:** a small vision-language model can follow instructions, stay on topic and produce clean structured output, but it should not be trusted for species identification. For a safety-critical domain like foraging, confident wrong answers are the biggest risk.

### Tech
`transformers` · `qwen-vl-utils` · `torch` (float16) · `gradio` · Kaggle GPU

---

## 2. Evaluating and Improving LLM Trustworthiness

**Goal:** Measure how trustworthy an open-weight LLM is across three dimensions, then reduce its biggest weakness.

**Setup:** Qwen3-4B, Hugging Face Transformers, half-precision inference on an NVIDIA T4 GPU (Google Colab), greedy decoding, and datasets from the [DecodingTrust](https://github.com/AI-secure/DecodingTrust) benchmark.

### Part A – Baseline evaluation

| Dimension | Dataset | Samples | Result |
|-----------|---------|---------|--------|
| **Common-sense ethics** | ETHICS `cm_test.csv` (short statements) | 100 | Accuracy **81%**, precision 91.4%, recall 66.7%, F1 77.1% |
| **Stereotype bias** | `user_prompts.csv` | 300 | **2 / 300** stereotypes endorsed (0.67% endorsement rate) |
| **Privacy** | PII password-attack prompts | 100 | Exact password disclosed in **92 / 100** responses |

- **Ethics:** most errors were false negatives (16 of 19). The model was more likely to call a wrong action acceptable than the other way round.
- **Bias:** the endorsement rate is low, but both failures were harmful health-related stereotypes, so a low rate is not the same as no risk.
- **Privacy:** the clearest weakness. When a prompt contained a list of passwords, the model almost always repeated the requested one.

### Part B – Privacy mitigation

| Metric | Baseline | Refusal policy (system prompt) | Output guard (filter) |
|--------|----------|--------------------------------|-----------------------|
| Exact disclosures | 92 / 100 | **0 / 100** | **0 / 100** |
| Disclosure rate | 92% | **0%** | **0%** |
| Refusal language detected | 0% | 100% | – |

- **Refusal policy:** a system instruction telling the model never to reveal or repeat credentials and to refuse politely.
- **Output guard:** a post-processing filter that blocks any response containing the target password.

**Limitations:** results come from small samples. The output guard already knows the target password, so it demonstrates exact-match filtering rather than a complete privacy system. Altered or partial passwords and other kinds of sensitive data were not tested.

📄 Full write-up: [`Trustworthiness_Evaluation/about_this_project`](Trustworthiness_Evaluation/about_this_project)

---

## Repository structure

```
LLMsUppsala/
├── chatbotmushroom.ipynb                  # Project 1 – multimodal mushroom chatbot
├── chatbot1.pdf                           # Project 1 – evaluation report
├── Trustworthiness_Evaluation/
│   ├── Assignment2_FinalVersion.ipynb     # Project 2 – code, results and report
│   └── about_this_project                 # Project 2 – detailed summary
└── README.md
```

## How to run

Both projects need a **GPU runtime** (an NVIDIA T4 on Google Colab or Kaggle is enough).

1. Open the notebook in Google Colab or Kaggle and select a GPU runtime.
2. Run the first cells to install dependencies, for example:
   ```bash
   pip install -U transformers accelerate qwen-vl-utils gradio
   ```
3. For Project 2, the notebook clones the DecodingTrust datasets automatically:
   ```bash
   git clone https://github.com/AI-secure/DecodingTrust.git
   ```
4. Run all cells in order. Project 1 prints a temporary public Gradio link where you can chat with the bot.

A Hugging Face token (`HF_TOKEN`) is optional but gives faster model downloads.

## Skills demonstrated

- Running and prompting open-weight LLMs and vision-language models with Hugging Face Transformers
- Prompt engineering: system prompts, topic control and structured JSON extraction
- Building interactive, streaming chat interfaces with Gradio
- Designing LLM evaluations: metrics, failure analysis and reproducibility
- AI safety and trustworthiness: ethics, bias, privacy leakage and mitigation

## References

1. Wang, B. et al. (2023). *DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models.* [decodingtrust.github.io](https://decodingtrust.github.io/)
2. Qwen Team. [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on Hugging Face.

---

## About me

**MD Nazmus Sadat Samin** – MSc student in Data Science with Machine Learning and Statistics at Uppsala University, interested in machine learning and natural language processing.

📫 sadatnazmus567@gmail.com · Stockholm, Sweden

*Academic coursework for 1RT730 Large Language Models, Uppsala University.*
