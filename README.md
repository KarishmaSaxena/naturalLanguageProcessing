# NLP Learning

This repository contains notes, examples, and small practicals for learning Natural Language Processing (NLP). The focus is on core tokenization and preprocessing techniques, basic terminology, classical encodings, and word embeddings (Word2Vec) including a practical implementation of averaging Word2Vec vectors.

## Contents

- Tokenization and basic terminologies
- Tokenization practicals / exercises
- Text preprocessing (Stemming and Lemmatization) using NLTK
- Stopwords, Parts of Speech (POS), Named Entity Recognition (NER)
- Different types of encoding (one-hot, Bag-of-Words, TF-IDF, embeddings)
- Word Embeddings, Word2Vec (CBOW and Skip-gram)
- Skip-gram: in-depth intuition
- Average Word2Vec with implementation

---

## Setup

Recommended Python packages:

```bash
pip install nltk spacy gensim scikit-learn
python -m spacy download en_core_web_sm
```

Also download common NLTK data (run once in a Python REPL):

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

(typos above should be corrected: use `nltk.download('wordnet')`, `nltk.download('averaged_perceptron_tagger')`, `nltk.download('stopwords')`)

---

## 1) Tokenization and Basic Terminologies

Tokenization is the process of splitting text into smaller units called tokens. Tokens can be words, subwords, or sentences.

- Word tokenization: splits into words/tokens (e.g., "I'm" -> ["I", "'m"] or ["I'm"] depending on tokenizer)
- Sentence tokenization: splits text into sentences
- Subword tokenization: BPE/WordPiece used in transformer models

NLTK example:

```python
from nltk.tokenize import word_tokenize, sent_tokenize
text = "Hello world! I'm learning NLP."
print(sent_tokenize(text))
print(word_tokenize(text))
```

Practical exercises:
- Tokenize a paragraph into sentences then words
- Compare results of simple split(" ") vs NLTK tokenizers
- Explore tokenization differences for contractions and punctuation

---

## 2) Tokenization Practicals

A few practical tasks to practice tokenization:
- Clean a text file by tokenizing and re-joining sentences by normalized spacing.
- Build a tokenizer that lowercases, removes punctuation, and returns words.
- Compare token counts for different tokenizers (NLTK, spaCy, simple whitespace).

---

## 3) Text Preprocessing

Common preprocessing steps:
- Lowercasing
- Remove/normalize punctuation
- Remove numbers or normalize them
- Remove extra whitespace
- Expand contractions (optional)

Example pipeline using NLTK and regex:

```python
import re
from nltk.tokenize import word_tokenize

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # keep only alphanumerics and spaces
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return tokens
```

---

## 4) Stemming using NLTK

Stemming reduces words to their root form using crude heuristics.

```python
from nltk.stem import PorterStemmer, LancasterStemmer
ps = PorterStemmer()
ls = LancasterStemmer()
words = ["running", "ran", "runs", "easily", "fairly"]
print([ps.stem(w) for w in words])
print([ls.stem(w) for w in words])
```

Notes: Stemmers are fast but may produce non-words and are language-dependent.

---

## 5) Lemmatization using NLTK

Lemmatization maps words to their dictionary base form using vocabulary and POS tags.

```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

wn = WordNetLemmatizer()

# helper to convert NLTK POS to WordNet POS

def nltk_pos_to_wordnet(nltk_pos):
    if nltk_pos.startswith('J'):
        return wordnet.ADJ
    elif nltk_pos.startswith('V'):
        return wordnet.VERB
    elif nltk_pos.startswith('N'):
        return wordnet.NOUN
    elif nltk_pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

text = "The striped bats are hanging on their feet for best"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
lemmas = [wn.lemmatize(tok, pos=nltk_pos_to_wordnet(pos)) for tok, pos in pos_tags]
print(lemmas)
```

Lemmatization generally produces real dictionary forms and is more accurate than stemming for many tasks.

---

## 6) Stopwords, POS, Named Entity Recognition (NER)

Stopwords: common words that often carry little semantic meaning (e.g., "the", "is").

```python
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
words = [w for w in tokens if w.lower() not in stop]
```

POS tagging: labeling words with parts of speech (noun, verb, adjective, ...). Use NLTK or spaCy for robust tagging.

```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_)

# NER
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## 7) Different Types of Encoding

- One-hot encoding: sparse binary vectors where each token is an index
- Bag-of-Words (CountVectorizer): counts of tokens
- TF-IDF: weighted counts reducing importance of frequent tokens
- Label encoding for categories
- Embeddings: dense continuous vectors capturing semantics

scikit-learn examples (CountVectorizer, TfidfVectorizer):

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
corpus = ["this is a sentence", "this is another sentence"]
cv = CountVectorizer()
print(cv.fit_transform(corpus).toarray())

tf = TfidfVectorizer()
print(tf.fit_transform(corpus).toarray())
```

---

## 8) Word Embedding, Word2Vec

Word embeddings map words to dense vectors. Word2Vec (Mikolov et al.) trains embeddings using either CBOW (predict target from context) or Skip-gram (predict context from target).

Gensim example (training a small Word2Vec model):

```python
from gensim.models import Word2Vec
sentences = ["this is a sentence".split(), "this is another sentence".split()]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2, sg=0)  # sg=0 -> CBOW
# sg=1 -> skip-gram
print(model.wv['sentence'])  # vector for the token 'sentence'
```

---

## 9) Skip-gram: In-depth Intuition

- Objective: Given a center (target) word, predict surrounding context words within a window.
- For each (target, context) pair, the model maximizes the probability of observing context words given the target word vector.
- Training uses many (target, context) pairs sampled from corpus. Negative sampling or hierarchical softmax are used to make training efficient.
- Skip-gram tends to do better for infrequent words, while CBOW is faster and works well for frequent words.

High-level math intuition (skip-gram with softmax):
- For target word w_t and context word w_c, model predicts:

  P(w_c | w_t) = exp(v'_c . v_t) / sum_{w in V} exp(v'_w . v_t)

  where v_t is the 'input' vector for the target and v'_c is the 'output' vector for the context.
- Negative sampling approximates this objective by contrasting positive pairs with noise samples.

---

## 10) Average Word2Vec (Implementation)

A common baseline: represent a document/sentence by the averaged word vectors of its tokens.

```python
import numpy as np

def average_word_vectors(tokens, model, vector_size):
    """Return the average Word2Vec vector for a list of tokens.
    tokens: list of strings
    model: gensim Word2Vec model or KeyedVectors
    vector_size: dimensionality of vectors
    """
    vecs = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
    if len(vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(vecs, axis=0)

# Example usage:
# vec = average_word_vectors(['this','is','a','sentence'], model, 100)
```

Notes:
- Handle out-of-vocabulary (OOV) words by skipping or using a random/zero vector
- Averaging loses word order and syntactic information but is a simple strong baseline for classification tasks

---

## Resources

- NLTK book: https://www.nltk.org/ 
- Gensim Word2Vec tutorial: https://radimrehurek.com/gensim/models/word2vec.html
- Original Word2Vec papers: Mikolov et al., 2013
- spaCy documentation: https://spacy.io/

---

## License

This content is for study and learning. Feel free to copy and adapt for your notes.