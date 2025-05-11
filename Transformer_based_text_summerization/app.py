import os
import requests
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, render_template, request

# -------------------------------
# Fetch Wikipedia Article Summary
# -------------------------------
url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

article_title = "Artificial_intelligence"

response = requests.get(url + article_title)


if response.status_code == 200:
    data = response.json()
    article_content = data.get('extract', 'No content found')
    print("Article fetched successfully!")
else:
    article_content = ""
    print(f"Error fetching article: {response.status_code}")

print(article_content)

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


# -------------------------------
# Summarize the Wikipedia Article
# -------------------------------
inputs = tokenizer(article_content, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Summary:")
print(summary)

# -------------------------------
# Flask Web Application
# -------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        user_content = request.form["article_content"]

        # Tokenize and summarize user content
        inputs = tokenizer(user_content, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":    
    app.run(debug=False, port=5000)




