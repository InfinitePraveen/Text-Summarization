from flask import Flask, render_template, request
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading summarization model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def split_into_chunks(text, max_tokens=850):
    """Split a long article into sentence-based chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        if not sentence.strip():
            continue

        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current and current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = token_count
        else:
            current.append(sentence)
            current_tokens += token_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def summarize_text(text):
    chunks = split_into_chunks(text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=150,
                min_length=35,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        summaries.append(
            tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        )

    return " ".join(summaries)


@app.route("/", methods=["GET", "POST"])
def index():
    article = ""
    summary = ""
    error = ""

    if request.method == "POST":
        article = request.form.get("article", "").strip()

        if len(article.split()) < 30:
            error = "Please enter an article with at least 30 words."
        else:
            try:
                summary = summarize_text(article)
            except Exception as exc:
                error = f"Could not generate the summary: {exc}"

    return render_template(
        "index.html",
        article=article,
        summary=summary,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
