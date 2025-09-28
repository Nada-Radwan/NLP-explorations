# 🧠 NLP Explorations with MALLET, Pattern & Polyglot
📌 Overview

This repository is a hands-on exploration of Natural Language Processing (NLP) using Python.
It demonstrates:

- MALLET → Topic modeling and text mining (via Python wrappers).

- Pattern → Web mining, POS tagging, sentiment analysis.

- Polyglot → Multilingual NLP: NER, POS, sentiment, and language detection.

- Core NLP techniques → tokenization, embeddings, transformers.

- Real-world NLP applications → classification, topic modeling, chatbots.

- NLP components → NLU (Natural Language Understanding) & NLG (Natural Language Generation).

✅ All code examples run smoothly on Google Colab.

## 📌 1. MALLET (MAchine Learning for LanguagE Toolkit)

MALLET is originally a Java-based package for **topic modeling**, **document classification**, and **sequence tagging**.  
In Python, we typically access MALLET through **Gensim** for **LDA (Latent Dirichlet Allocation)** topic modeling.

- Gensim → A popular Python library for natural language processing (NLP) and topic modeling.

- LDA (Latent Dirichlet Allocation) → A statistical method used to discover hidden topics in a collection of documents.
  
### Example: Topic Modeling with MALLET via Gensim
```python
!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
!unzip -q mallet-2.0.8.zip

from gensim.models.wrappers import LdaMallet
from gensim import corpora

# Sample corpus
texts = [
    ["nlp", "is", "fun"],
    ["machine", "learning", "and", "nlp"],
    ["topic", "modeling", "with", "mallet"]
]

# Create dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train MALLET LDA model
mallet_path = './mallet-2.0.8/bin/mallet'
ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=2, id2word=dictionary)

print(ldamallet.show_topics()) 
```
## 📌 2. Pattern

Pattern is a Python package for:

- Web mining

- Natural language processing

- Machine learning

It includes tools for POS tagging, sentiment analysis, and vector space modeling.

Example: Sentiment Analysis with Pattern

```python
!pip install pattern

from pattern.en import sentiment

print(sentiment("I love Natural Language Processing!"))
print(sentiment("This project is terrible."))
```
## 📌 3. Polyglot

Polyglot is a multilingual NLP library.
It supports named entity recognition (NER), language detection, sentiment analysis, and more in over 130 languages.

Example: Named Entity Recognition with Polyglot
```python
!pip install polyglot pyicu pycld2 morfessor

from polyglot.text import Text

txt = Text("Barack Obama was born in Hawaii. He was elected president in 2008.")
print(txt.entities)
```
## 📌 4. NLP Techniques

Some key NLP techniques explored in this repo:

- Tokenization – splitting text into words or sentences

- Stemming & Lemmatization – reducing words to their base form

- Part-of-Speech (POS) Tagging

- Named Entity Recognition (NER)

- Sentiment Analysis

- Topic Modeling (LDA, MALLET)

- Word Embeddings (Word2Vec, GloVe, FastText)
  
## 📌 5. NLP Applications

Real-world applications of NLP:

- Chatbots and Virtual Assistants (e.g., Siri, Alexa)

- Machine Translation (Google Translate, DeepL)

- Text Classification (spam detection, news categorization)

- Information Retrieval (search engines)

- Sentiment Analysis (social media monitoring)

- Speech Recognition (voice commands)

## 📌 6. NLP Components: NLU vs NLG

- NLU (Natural Language Understanding):
Focuses on extracting meaning from text.
Examples: intent detection, sentiment analysis, entity recognition.
```python
def parse_intent(text):
    if "book" in text.lower():
        return "BookFlight"
    return "Unknown"

print(parse_intent("Book me a flight to London"))
```
- NLG (Natural Language Generation):
Focuses on generating human-like language.
Examples: text summarization, chatbot responses, story generation.
```python
def generate_summary(name, role):
    return f"{name} works as a {role}."

print(generate_summary("Alice", "Data Scientist"))
```
