from flask import Flask, render_template, request
import os
import re
import nltk
import pytesseract

from PIL import Image
from collections import Counter
from transformers import pipeline

# =========================================
# DOWNLOAD NLTK
# =========================================

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================
# LOAD SENTIMENT MODEL
# =========================================

sentiment_pipeline = pipeline(
    "sentiment-analysis"
)

# =========================================
# POSITIVE / NEGATIVE WORDS
# =========================================

positive_keywords = [

    "good",
    "great",
    "excellent",
    "awesome",
    "amazing",
    "comfortable",
    "premium",
    "fast",
    "smooth",
    "clear",
    "best",
    "love",
    "perfect"

]

negative_keywords = [

    "bad",
    "worst",
    "issue",
    "problem",
    "poor",
    "disappointed",
    "heating",
    "lag",
    "slow",
    "waste",
    "broken",
    "damage"

]

# =========================================
# HOME ROUTE
# =========================================

@app.route("/", methods=["GET", "POST"])

def home():

    overall_sentiment = ""
    confidence = ""
    recommendation = ""

    positive_sentences = []
    negative_sentences = []

    pros = []
    cons = []

    if request.method == "POST":

        files = request.files.getlist("images")

        combined_reviews = ""

        # =========================================
        # OCR FROM MULTIPLE IMAGES
        # =========================================

        for file in files:

            if file.filename != "":

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    file.filename
                )

                file.save(filepath)

                image = Image.open(filepath)

                extracted_text = pytesseract.image_to_string(
                    image
                )

                combined_reviews += extracted_text + " "

        # =========================================
        # CLEAN TEXT
        # =========================================

        combined_reviews = re.sub(
            r'\s+',
            ' ',
            combined_reviews
        )

        # =========================================
        # OVERALL SENTIMENT
        # =========================================

        result = sentiment_pipeline(
            combined_reviews[:512]
        )

        overall_sentiment = result[0]['label']

        confidence = round(
            result[0]['score'] * 100,
            2
        )

        # =========================================
        # SENTENCE ANALYSIS
        # =========================================

        sentences = sent_tokenize(
            combined_reviews
        )

        for sentence in sentences:

            try:

                analysis = sentiment_pipeline(
                    sentence[:512]
                )

                label = analysis[0]['label']

                if label == "POSITIVE":

                    positive_sentences.append(
                        sentence
                    )

                else:

                    negative_sentences.append(
                        sentence
                    )

            except:
                pass

        # =========================================
        # EXTRACT PROS
        # =========================================

        positive_text = " ".join(
            positive_sentences
        ).lower()

        for word in positive_keywords:

            if word in positive_text:

                pros.append(word)

        # =========================================
        # EXTRACT CONS
        # =========================================

        negative_text = " ".join(
            negative_sentences
        ).lower()

        for word in negative_keywords:

            if word in negative_text:

                cons.append(word)

        # =========================================
        # FINAL RECOMMENDATION
        # =========================================

        if overall_sentiment == "POSITIVE":

            recommendation = "Worth Buying ✅"

        else:

            recommendation = "Not Recommended ❌"

    # =========================================
    # RETURN RESULT
    # =========================================

    return render_template(

        "index.html",

        overall_sentiment=overall_sentiment,

        confidence=confidence,

        recommendation=recommendation,

        pros=pros,

        cons=cons,

        positive_sentences=positive_sentences[:5],

        negative_sentences=negative_sentences[:5]

    )

# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )