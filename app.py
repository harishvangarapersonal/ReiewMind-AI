from flask import Flask, render_template, request

import easyocr

from transformers import pipeline

import nltk

from nltk.tokenize import sent_tokenize

import os

# =========================================
# DOWNLOAD NLTK DATA
# =========================================

nltk.download('punkt')

nltk.download('punkt_tab')

# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)

# =========================================
# UPLOAD FOLDER
# =========================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder automatically
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# =========================================
# OCR MODEL
# =========================================

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# =========================================
# SENTIMENT MODEL
# =========================================

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# =========================================
# HOME ROUTE
# =========================================

@app.route("/", methods=["GET", "POST"])

def home():

    overall_sentiment = ""

    confidence = ""

    recommendation = ""

    rating = ""

    positive_sentences = []

    negative_sentences = []

    extracted_reviews = ""

    if request.method == "POST":

        files = request.files.getlist(
            "images"
        )

        all_reviews = []

        # ====================================
        # OCR FROM MULTIPLE IMAGES
        # ====================================

        for file in files:

            if file.filename == "":

                continue

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            try:

                file.save(filepath)

            except:

                continue

            # OCR
            try:

                result = reader.readtext(
                    filepath,
                    detail=0
                )

            except:

                continue

            text = " ".join(result)

            all_reviews.append(text)

        # ====================================
        # HANDLE EMPTY REVIEWS
        # ====================================

        if len(all_reviews) == 0:

            return render_template(
                "index.html",
                overall_sentiment="No readable text found",
                confidence=0,
                recommendation="Upload clearer screenshots",
                rating=0,
                positive_sentences=[],
                negative_sentences=[],
                extracted_reviews=""
            )

        # ====================================
        # COMBINE REVIEWS
        # ====================================

        combined_reviews = " ".join(
            all_reviews
        )

        extracted_reviews = combined_reviews

        # ====================================
        # OVERALL SENTIMENT
        # ====================================

        try:

            sentiment = sentiment_pipeline(
                combined_reviews[:512]
            )

            overall_sentiment = sentiment[0]['label']

            confidence = round(
                sentiment[0]['score'] * 100,
                2
            )

        except:

            overall_sentiment = "UNKNOWN"

            confidence = 0

        # ====================================
        # SPLIT INTO SENTENCES
        # ====================================

        try:

            sentences = sent_tokenize(
                combined_reviews
            )

        except:

            sentences = combined_reviews.split(".")

        # ====================================
        # ANALYZE EACH SENTENCE
        # ====================================

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 4:

                continue

            try:

                result = sentiment_pipeline(
                    sentence[:512]
                )

            except:

                continue

            label = result[0]['label']

            if label == "POSITIVE":

                positive_sentences.append(
                    sentence
                )

            else:

                negative_sentences.append(
                    sentence
                )

        # ====================================
        # PRODUCT RATING
        # ====================================

        positive_count = len(
            positive_sentences
        )

        negative_count = len(
            negative_sentences
        )

        total = (
            positive_count +
            negative_count
        )

        if total == 0:

            rating = 3.0

        else:

            positivity_ratio = (
                positive_count / total
            )

            rating = round(
                positivity_ratio * 5,
                1
            )

        # ====================================
        # FINAL RECOMMENDATION
        # ====================================

        if positive_count >= negative_count:

            recommendation = (
                "Worth Buying ✅"
            )

        else:

            recommendation = (
                "Not Recommended ❌"
            )

    return render_template(
        "index.html",
        overall_sentiment=overall_sentiment,
        confidence=confidence,
        recommendation=recommendation,
        rating=rating,
        positive_sentences=positive_sentences,
        negative_sentences=negative_sentences,
        extracted_reviews=extracted_reviews
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