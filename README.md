# 🧠 NLP Explorations with MALLET, Pattern & Polyglot
📌 Overview

This repository is a hands-on exploration of Natural Language Processing (NLP) using Python.
It demonstrates:

MALLET → Topic modeling and text mining (via Python wrappers).

Pattern → Web mining, POS tagging, sentiment analysis.

Polyglot → Multilingual NLP: NER, POS, sentiment, and language detection.

Core NLP techniques → tokenization, embeddings, transformers.

Real-world NLP applications → classification, topic modeling, chatbots.

NLP components → NLU (Natural Language Understanding) & NLG (Natural Language Generation).

✅ All code examples run smoothly on Google Colab.

🚀 Getting Started
🔧 Requirements

Python 3.8+

Google Colab (recommended) or local Jupyter Notebook

📥 Installation (Colab)

# Install dependencies
!apt-get install -y python3-dev libicu-dev
!pip install pattern polyglot pyicu pycld2 morfessor transformers

# MALLET setup (Colab):
!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
!unzip -q mallet-2.0.8.zip

📘 Examples
🔹 MALLET (Topic Modeling)

import subprocess

mallet_bin = "/content/mallet-2.0.8/bin/mallet"

# Import data
subprocess.run([mallet_bin, "import-dir", "--input", "data/texts",
                "--output", "corpus.mallet", "--keep-sequence", "--remove-stopwords"])

# Train topics
subprocess.run([mallet_bin, "train-topics", "--input", "corpus.mallet",
                "--num-topics", "5", "--output-topic-keys", "topics.txt"])

