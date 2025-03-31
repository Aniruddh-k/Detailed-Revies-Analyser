import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import tempfile
import textwrap
import base64

# --- Styling ---
st.markdown(
    """
    <style>
    .reportview-container {
        margin-top: -2em;
    }
    .st-header {
        padding-bottom: 0;
    }
    .st-subheader {
        margin-top: 1em;
        color: #555;
    }
    .st-file-uploader {
        margin-bottom: 2em;
    }
    .st-button {
        margin-top: 1em;
    }
    .report-section {
        padding: 1em;
        margin-bottom: 1.5em;
        border-radius: 5px;
        background-color: #f9f9f9;
        border: 1px solid #eee;
    }
    .report-item {
        margin-bottom: 0.5em;
        color: #666;
    }
    .report-image {
        margin-top: 1em;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .stDownloadButton {
        margin-top: 2em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Models ---
@st.cache_resource
def load_models():
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer_bert = DistilBertTokenizer.from_pretrained(model_name)
    model_bert = DistilBertForSequenceClassification.from_pretrained(model_name)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return tokenizer_bert, model_bert, summarizer

tokenizer_bert, model_bert, summarizer = load_models()

# --- Sentiment Prediction ---
def predict_sentiment(text):
    inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model_bert(**inputs).logits
    predictions = torch.argmax(logits, dim=-1).item()
    return predictions

# --- Generate Plots ---
def generate_plots(report):
    positive_percentage = report['positive_percentage']
    negative_percentage = report['negative_percentage']

    fig1, ax1 = plt.subplots()
    ax1.pie([positive_percentage, negative_percentage], labels=['Positive', 'Negative'], autopct='%1.1f%%',
            startangle=90, colors=['#66b3ff', '#ff6666'])
    ax1.axis('equal')
    pie_buffer = BytesIO()
    plt.savefig(pie_buffer, format='png')
    pie_buffer.seek(0)
    plt.close(fig1)

    if report['top_positive_keywords']:
        top_pos_keywords, top_pos_counts = zip(*report['top_positive_keywords'])
        fig2, ax2 = plt.subplots()
        ax2.bar(top_pos_keywords, top_pos_counts, color='#66b3ff')
        ax2.set_xlabel('Keywords')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Top Positive Keywords')
        bar_buffer = BytesIO()
        plt.savefig(bar_buffer, format='png')
        bar_buffer.seek(0)
        plt.close(fig2)
        return pie_buffer, bar_buffer
    else:
        return pie_buffer, None

# --- Generate Summary ---
def generate_summary_with_bart(report):
    report_text = f"Sentiment Analysis Report:\n"
    report_text += f"Total Reviews: {report['total_reviews']}\n"
    report_text += f"Positive Sentiment: {report['positive_percentage']:.2f}%\n"
    report_text += f"Negative Sentiment: {report['negative_percentage']:.2f}%\n"
    report_text += f"Positive Count: {report['positive_count']}\n"
    report_text += f"Negative Count: {report['negative_count']}\n"
    report_text += f"\nMost Frequent Positive Texts:\n"
    for text, _ in report['most_frequent_positive']:
        report_text += f"- {text}\n"
    report_text += f"\nMost Frequent Negative Texts:\n"
    for text, _ in report['most_frequent_negative']:
        report_text += f"- {text}\n"
    report_text += f"\nTop Positive Keywords:\n"
    for keyword, count in report['top_positive_keywords']:
        report_text += f"{keyword}: {count}\n"
    report_text += f"\nTop Negative Keywords:\n"
    for keyword, count in report['top_negative_keywords']:
        report_text += f"{keyword}: {count}\n"

    inputs = summarizer.tokenizer(report_text, truncation=True, max_length=1024, return_tensors="pt")
    summary = summarizer.model.generate(inputs['input_ids'], max_length=250, min_length=100, num_beams=4, early_stopping=True)
    summary_text = summarizer.tokenizer.decode(summary[0], skip_special_tokens=True)

    return summary_text

# --- Generate PDF Report ---
def generate_pdf_with_summary(report, summary, pie_buffer, bar_buffer):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(30, 750, "Sentiment Analysis Report")
    c.drawString(30, 730, f"Total Reviews: {report['total_reviews']}")
    c.drawString(30, 710, f"Positive Sentiment: {report['positive_percentage']:.2f}%")
    c.drawString(30, 690, f"Negative Sentiment: {report['negative_percentage']:.2f}%")
    c.drawString(30, 670, f"Positive Count: {report['positive_count']}")
    c.drawString(30, 650, f"Negative Count: {report['negative_count']}")

    y_position = 630
    c.drawString(30, y_position, "Most Frequent Positive Texts:")
    y_position -= 20
    for i, (text, _) in enumerate(report['most_frequent_positive']):
        wrapped_lines = textwrap.wrap(f"{i+1}. {text}", width=90)
        for line in wrapped_lines:
            c.drawString(30, y_position, line)
            y_position -= 15
        y_position -= 5

    y_position -= 10
    c.drawString(30, y_position, "Most Frequent Negative Texts:")
    y_position -= 20
    for i, (text, _) in enumerate(report['most_frequent_negative']):
        wrapped_lines = textwrap.wrap(f"{i+1}. {text}", width=90)
        for line in wrapped_lines:
            c.drawString(30, y_position, line)
            y_position -= 15
        y_position -= 5

    if pie_buffer:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_pie:
            tmp_pie.write(pie_buffer.getvalue())
            pie_temp_path = tmp_pie.name
        c.drawImage(pie_temp_path, 30, 300, width=200, height=200)
        os.remove(pie_temp_path)

    if bar_buffer:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_bar:
            tmp_bar.write(bar_buffer.getvalue())
            bar_temp_path = tmp_bar.name
        c.drawImage(bar_temp_path, 250, 300, width=200, height=200)
        os.remove(bar_temp_path)

    y_position = 280
    c.drawString(30, y_position, "Generated Summary:")
    y_position -= 20
    for line in summary.splitlines():
        wrapped_lines = textwrap.wrap(line, width=90)
        for wrapped_line in wrapped_lines:
            c.drawString(30, y_position, wrapped_line)
            y_position -= 15

    c.save()
    buffer.seek(0)
    return buffer

# --- Main Streamlit App ---
def main():
    st.title("✨ Detailed Reviews Analyser ✨")
    st.subheader("Upload a CSV file to analyze sentiment and generate a detailed report.")
    st.text("Please add a CSV in which the reviews refer to only a single product for optimum results")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            if st.button("🚀 Analyze Sentiment and Generate Report"):
                if df.empty:
                    st.error("⚠️ Uploaded CSV is empty.")
                elif df.shape[1] < 1:
                    st.error("⚠️ CSV must have at least one column containing text data.")
                else:
                    with st.spinner("🧠 Analyzing sentiment and generating report..."):
                        texts = df.iloc[:, 0].values
                        predictions = [predict_sentiment(text) for text in texts]
                        df['predictions'] = predictions

                        sentiment_counts = df['predictions'].value_counts()
                        positive_percentage = (sentiment_counts.get(1, 0) / len(df)) * 100
                        negative_percentage = (sentiment_counts.get(0, 0) / len(df)) * 100

                        positive_texts = df[df['predictions'] == 1].iloc[:, 0].values
                        negative_texts = df[df['predictions'] == 0].iloc[:, 0].values

                        most_frequent_positive = Counter(positive_texts).most_common(5)
                        most_frequent_negative = Counter(negative_texts).most_common(5)

                        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10)
                        vectorizer.fit(positive_texts.tolist() + negative_texts.tolist())
                        positive_ngrams = vectorizer.transform(positive_texts.tolist()).toarray()
                        negative_ngrams = vectorizer.transform(negative_texts.tolist()).toarray()

                        positive_keywords = vectorizer.get_feature_names_out()
                        positive_word_counts = positive_ngrams.sum(axis=0)
                        negative_word_counts = negative_ngrams.sum(axis=0)

                        positive_keyword_freq = dict(zip(positive_keywords, positive_word_counts))
                        negative_keyword_freq = dict(zip(positive_keywords, negative_word_counts))

                        report = {
                            'total_reviews': len(df),
                            'positive_percentage': positive_percentage,
                            'negative_percentage': negative_percentage,
                            'positive_count': sentiment_counts.get(1, 0),
                            'negative_count': sentiment_counts.get(0, 0),
                            'most_frequent_positive': most_frequent_positive,
                            'most_frequent_negative': most_frequent_negative,
                            'top_positive_keywords': sorted(positive_keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5],
                            'top_negative_keywords': sorted(negative_keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5],
                        }

                        pie_buffer, bar_buffer = generate_plots(report)
                        summary = generate_summary_with_bart(report)
                        pdf_buffer = generate_pdf_with_summary(report, summary, pie_buffer, bar_buffer)

                        st.subheader("📊 Sentiment Analysis Results")

                        with st.container(border=True):
                            st.markdown("Overall Sentiment", unsafe_allow_html=False)
                            st.markdown(f"<p class='report-item'>Total Reviews: <b>{report['total_reviews']}</b></p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='report-item'>Positive Sentiment: <b>{report['positive_percentage']:.2f}%</b> ({report['positive_count']})</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='report-item'>Negative Sentiment: <b>{report['negative_percentage']:.2f}%</b> ({report['negative_count']})</p>", unsafe_allow_html=True)
                            if pie_buffer:
                                st.image(pie_buffer, caption="Sentiment Distribution", use_column_width=True, output_format="PNG", clamp=True, channels="RGB", width=400)

                        with st.container(border=True):
                            st.markdown("Most Frequent Reviews", unsafe_allow_html=True)
                            st.subheader("Top 5 Positive Reviews", divider=True)
                            for text, count in report['most_frequent_positive']:
                                st.markdown(f"<p class='report-item'>- {text}</p>", unsafe_allow_html=True)
                            st.subheader("Top 5 Negative Reviews", divider=True)
                            for text, count in report['most_frequent_negative']:
                                st.markdown(f"<p class='report-item'>- {text}</p>", unsafe_allow_html=True)

                        with st.container(border=True):
                            st.markdown("Key Themes", unsafe_allow_html=True)
                            st.subheader("Top 5 Positive Keywords", divider=True)
                            if report['top_positive_keywords']:
                                for keyword, count in report['top_positive_keywords']:
                                    st.markdown(f"<p class='report-item'>- {keyword}: <b>{count}</b></p>", unsafe_allow_html=True)
                                if bar_buffer:
                                    st.image(bar_buffer, caption="Top Positive Keywords Frequency", use_column_width=True, output_format="PNG", clamp=True, channels="RGB", width=600)
                            else:
                                st.info("No significant positive keywords found.")

                            st.subheader("Top 5 Negative Keywords", divider=True)
                            if report['top_negative_keywords']:
                                for keyword, count in report['top_negative_keywords']:
                                    st.markdown(f"<p class='report-item'>- {keyword}: <b>{count}</b></p>", unsafe_allow_html=True)
                            else:
                                st.info("No significant negative keywords found.")

                        with st.container(border=True):
                            st.markdown("Summary", unsafe_allow_html=True)
                            st.info(summary)

                        st.download_button(
                            label="📄 Download Full Report as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name="Sentiment_Analysis_Report.pdf",
                            mime="application/pdf",
                        )

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

if __name__ == "__main__":
    main()