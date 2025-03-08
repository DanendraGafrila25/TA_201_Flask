from flask import Flask, render_template, request
import nltk
import spacy
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk import word_tokenize

# Download resources yang dibutuhkan
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Muat model spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Fungsi untuk mengkonversi POS Tag ke format WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

# Fungsi untuk pra-pemrosesan teks
def preprocess_text(text):
    text = re.sub(r'\biam\b', 'i am', text, flags=re.IGNORECASE)
    text = re.sub(r'\bits\b', 'it is', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk menghitung kesamaan

def calculate_similarity(answer_text, criteria_text):
    answer_text = preprocess_text(answer_text)
    answer_doc = nlp(answer_text)
    criteria_doc = nlp(criteria_text)
    spacy_similarity = answer_doc.similarity(criteria_doc)

    answer_keywords = [(token.lemma_.lower(), any(child.dep_ == 'neg' for child in token.children))
                       for token in answer_doc if not token.is_stop and not token.is_punct]

    criteria_keywords = set([token.lemma_.lower() for token in criteria_doc
                              if not token.is_stop and not token.is_punct])

    matches = sum(1 if not is_negated else -0.5
                  for keyword, is_negated in answer_keywords if keyword in criteria_keywords)

    keyword_ratio = matches / len(criteria_keywords) if criteria_keywords else 0
    if keyword_ratio < 0:
        keyword_ratio = 0

    combined_score = (spacy_similarity + keyword_ratio) / 2
    return combined_score

criteria = {
    1: "Did not collect any data at all",
    2: "Collected a small portion of the data, incomplete",
    3: "Data has been collected, but there are some deficiencies",
    4: "Data collected is quite good and meets the requirements",
    5: "Data collected is perfect and has been validated"
}

question = "Have you collected the data according to the requirements?"

@app.route('/')
def index():
    return render_template('index.html', question=question, criteria=criteria)

@app.route('/assess', methods=['POST'])
def assess():
    answer = request.form['answer']
    score_given = int(request.form['score'])

    preprocessed_answer = preprocess_text(answer)
    doc_answer = nlp(preprocessed_answer)

    pos_tags = [(token.text, token.tag_) for token in doc_answer]
    total_pos = total_neg = count = 0

    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        if wn_pos:
            synset = lesk(preprocessed_answer, word, wn_pos)
            if synset:
                try:
                    swn_synset = swn.senti_synset(synset.name())
                    total_pos += swn_synset.pos_score()
                    total_neg += swn_synset.neg_score()
                    count += 1
                except:
                    continue

    avg_pos = total_pos / count if count > 0 else 0
    avg_neg = total_neg / count if count > 0 else 0

    sentiment = "Positive" if avg_pos > avg_neg else "Negative" if avg_neg > avg_pos else "Neutral"

    similarity_scores = {score: calculate_similarity(preprocessed_answer, description)
                         for score, description in criteria.items()}

    best_score = max(similarity_scores, key=similarity_scores.get)
    best_similarity = similarity_scores[best_score]

    return render_template('result.html', answer=answer, score_given=score_given,
                           sentiment=sentiment, avg_pos=avg_pos, avg_neg=avg_neg,
                           best_score=best_score, best_similarity=f"{best_similarity:.4f}")

if __name__ == '__main__':
    app.run(debug=True)