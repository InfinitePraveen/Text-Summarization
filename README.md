# Text Summarization

An interview-ready NLP project that generates concise summaries from long articles using a pretrained **BART Transformer** model from Hugging Face.

## Highlights

- Abstractive text summarization
- Hugging Face Transformers
- BART fine-tuned on CNN/DailyMail
- CNN/DailyMail dataset exploration
- Tokenization and Transformer inference
- ROUGE evaluation
- Long-article chunking
- Flask web application
- GitHub and LinkedIn links inside the web app
- Simple human-readable repository structure
- No `src/`, preprocessing package, or separate utility modules

## Dataset

The project uses the **CNN/DailyMail** dataset through the Hugging Face `datasets` library.

## Model

`facebook/bart-large-cnn`

This checkpoint is BART fine-tuned for CNN/DailyMail summarization.

> The project pins `transformers<5` because the classic `pipeline("summarization")` API changed in Transformers 5. Direct tokenizer/model loading is used here so the implementation remains clear and easy to understand.

## Repository Structure

```text
Text-Summarization/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_bart_text_summarization.ipynb
│   └── 03_summarization_evaluation.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── .gitignore
```

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run the Notebooks

```bash
jupyter notebook
```

Run them in this order:

1. `01_dataset_exploration.ipynb`
2. `02_bart_text_summarization.ipynb`
3. `03_summarization_evaluation.ipynb`

The first model run downloads the tokenizer and model from Hugging Face.

## Run the Flask Web App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Paste an article and click **Summarize Article**.

The application splits very long articles into sentence-aware chunks, summarizes each chunk, and combines the generated summaries.

## Skills Demonstrated

- NLP
- Transformers
- Hugging Face
- BART
- Sequence-to-sequence models
- Tokenization
- Text generation
- Abstractive summarization
- ROUGE
- Dataset exploration
- Python
- Flask
- HTML/CSS
- Model deployment

## Interview Topics

Be prepared to explain:

1. Extractive vs. abstractive summarization
2. Why BART is useful for summarization
3. Encoder-decoder Transformers
4. Self-attention and cross-attention
5. Tokenization
6. Maximum input length
7. Long-document chunking
8. Beam search
9. `min_length` and `max_length`
10. ROUGE-1, ROUGE-2 and ROUGE-L
11. Hallucination in generative NLP
12. Fine-tuning vs. inference
13. CNN/DailyMail
14. Loading a Transformer in Flask
15. CPU vs. GPU inference

## Author

**Praveen Kumar**

Data Scientist | Open Source Learner | IBM Certified

GitHub: https://github.com/InfinitePraveen

LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## License

Educational and portfolio use.
