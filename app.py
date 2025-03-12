from flask import Flask, render_template, request, jsonify
import nltk
import spacy
import re
import logging
import json
import os
from datetime import datetime
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk import word_tokenize
from translate import Translator

# Konfigurasi logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('flask_app')

# Download resources yang dibutuhkan
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Muat model spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize translator
translator = Translator(to_lang="en", from_lang="id")

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

# Fungsi untuk menerjemahkan teks
def translate_text(text):
    try:
        translated_text = translator.translate(text)
        logger.info(f"Translated text: {translated_text}")
        return translated_text
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text  # Return original text if translation fails

# Fungsi untuk menghitung kesamaan
def calculate_similarity(answer_text, criteria_text):
    # Clean and normalize criteria text (remove quotes, extra newlines, etc.)
    criteria_text = re.sub(r'[""]', '', criteria_text)  # Remove quotes
    criteria_text = re.sub(r'\n+', ' ', criteria_text)  # Replace newlines with spaces
    criteria_text = criteria_text.strip()
    
    logger.info(f"Calculating similarity between: \nAnswer: '{answer_text}'\nCriteria: '{criteria_text}'")
    
    answer_text = preprocess_text(answer_text)
    answer_doc = nlp(answer_text)
    criteria_doc = nlp(criteria_text)
    
    try:
        spacy_similarity = answer_doc.similarity(criteria_doc)
        logger.info(f"spaCy similarity score: {spacy_similarity}")
    except Warning as w:
        logger.warning(f"Warning during similarity calculation: {w}")
        spacy_similarity = 0.0
    except Exception as e:
        logger.error(f"Error during similarity calculation: {e}")
        spacy_similarity = 0.0

    answer_keywords = [(token.lemma_.lower(), any(child.dep_ == 'neg' for child in token.children))
                       for token in answer_doc if not token.is_stop and not token.is_punct]

    criteria_keywords = set([token.lemma_.lower() for token in criteria_doc
                              if not token.is_stop and not token.is_punct])
    
    logger.info(f"Answer keywords: {answer_keywords}")
    logger.info(f"Criteria keywords: {criteria_keywords}")

    matches = sum(1 if not is_negated else -0.5
                  for keyword, is_negated in answer_keywords if keyword in criteria_keywords)
    
    logger.info(f"Keyword matches: {matches}")

    keyword_ratio = matches / len(criteria_keywords) if criteria_keywords else 0
    if keyword_ratio < 0:
        keyword_ratio = 0
    
    logger.info(f"Keyword ratio: {keyword_ratio}")

    combined_score = (spacy_similarity + keyword_ratio) / 2
    logger.info(f"Combined similarity score: {combined_score}")
    
    return combined_score

# Default criteria for web interface
default_criteria = {
    1: "Did not collect any data at all",
    2: "Collected a small portion of the data, incomplete",
    3: "Data has been collected, but there are some deficiencies",
    4: "Data collected is quite good and meets the requirements",
    5: "Data collected is perfect and has been validated"
}

default_question = "Have you collected the data according to the requirements?"

@app.route('/')
def index():
    return render_template('index.html', question=default_question, criteria=default_criteria)

# Function to load criteria from Rubrik.json based on question_id
def load_criteria_from_rubrik(question_id):
    try:
        rubrik_file = "Rubrik.json"
        if not os.path.exists(rubrik_file):
            logger.warning(f"Rubrik file not found: {rubrik_file}")
            return None
        
        with open(rubrik_file, 'r', encoding='utf-8') as f:
            rubrik_data = json.load(f)
        
        # Match based on question_id which corresponds to assessment_id in Rubrik.json
        for assessment_id, assessment_data in rubrik_data.items():
            if assessment_id == question_id:
                logger.info(f"Found matching criteria for question_id: {question_id}")
                return assessment_data
        
        logger.warning(f"No matching criteria found for question_id: {question_id}")
        return None
    except Exception as e:
        logger.error(f"Error loading criteria from Rubrik.json: {e}")
        return None

# New route for import-criteria remains unchanged
@app.route('/import-criteria', methods=['POST'])
def import_criteria():
    if request.is_json:
        data = request.json
        logger.info(f"Received criteria import data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        try:
            assessment_id = data.get('assessment_id')
            rubrik_id = data.get('rubrik_id')
            question = data.get('question', '')
            
            # Create criteria dictionary from bobot_1 to bobot_5
            raw_criteria = {
                1: data.get('bobot_1', ''),
                2: data.get('bobot_2', ''),
                3: data.get('bobot_3', ''),
                4: data.get('bobot_4', ''),
                5: data.get('bobot_5', '')
            }
            
            # Filter out empty criteria
            criteria = {k: v for k, v in raw_criteria.items() if v and v.strip()}
            
            # Translate criteria to English
            translated_criteria = {}
            for score, description in criteria.items():
                translated_description = translate_text(description)
                translated_criteria[score] = translated_description
            
            # Log the translated criteria
            logger.info(f"Translated criteria for assessment_id {assessment_id}: {json.dumps(translated_criteria, indent=2, ensure_ascii=False)}")
            
            # Save criteria to JSON file
            rubrik_file = "Rubrik.json"
            
            # Load existing data if file exists
            try:
                if os.path.exists(rubrik_file):
                    with open(rubrik_file, 'r', encoding='utf-8') as f:
                        rubrik_data = json.load(f)
                else:
                    rubrik_data = {}
            except Exception as e:
                logger.error(f"Error loading existing rubrik data: {e}")
                rubrik_data = {}
            
            # Add or update criteria for this assessment_id
            rubrik_data[str(assessment_id)] = {
                "assessment_id": assessment_id,
                "rubrik_id": rubrik_id,
                "question": question,
                "aspect": data.get('aspect', ''),
                "criteria": data.get('criteria', ''),
                "skill_type": data.get('skill_type', ''),
                "type": data.get('type', ''),
                "raw_criteria": {str(k): v for k, v in criteria.items()},
                "translated_criteria": {str(k): v for k, v in translated_criteria.items()},
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Write updated data back to file
            with open(rubrik_file, 'w', encoding='utf-8') as f:
                json.dump(rubrik_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully saved criteria for assessment_id {assessment_id} to {rubrik_file}")
            
            return jsonify({
                'success': True,
                'assessment_id': assessment_id,
                'question': question,
                'translated_criteria': translated_criteria,
                'message': f"Data berhasil disimpan ke dalam {rubrik_file}"
            })
            
        except Exception as e:
            logger.error(f"Error processing criteria import: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    else:
        return jsonify({
            'success': False,
            'error': 'Invalid request format, JSON expected'
        }), 400

@app.route('/assess', methods=['POST'])
def assess():
    # Check if the request is from a web form or an API call
    if request.is_json:
        # API request from Laravel
        data = request.json
        
        # Log data yang diterima dari Laravel
        logger.info(f"Data diterima dari Laravel: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # Log headers untuk debugging koneksi
        logger.info(f"Request Headers: {dict(request.headers)}")
        
        # Extract the simplified data
        question_id = data.get('question_id')
        answer = data.get('answer', '')
        score_given = int(data.get('score', 1))
        
        # Load criteria from Rubrik.json based on question_id
        assessment_data = load_criteria_from_rubrik(question_id)
        
        if assessment_data:
            # Use the translated criteria from Rubrik.json
            translated_criteria = {}
            for score, description in assessment_data.get('translated_criteria', {}).items():
                translated_criteria[int(score)] = description
            
            logger.info(f"Loaded criteria for question_id {question_id}: {json.dumps(translated_criteria, indent=2, ensure_ascii=False)}")
        else:
            # Use default criteria if no match found
            translated_criteria = default_criteria
            logger.warning(f"Using default criteria for question_id {question_id}")
        
        # Ensure we have criteria for all scores 1-5
        for i in range(1, 6):
            if i not in translated_criteria:
                translated_criteria[i] = default_criteria[i]
                logger.warning(f"Missing criteria for score {i}, using default")
    else:
        # Web form request
        answer = request.form['answer']
        score_given = int(request.form['score'])
        translated_criteria = default_criteria
        logger.info(f"Data diterima dari Web Form: answer={answer}, score={score_given}")

    # Translate the answer to English
    try:
        translated_answer = translate_text(answer)
        logger.info(f"Translated answer: {translated_answer}")
    except Exception as e:
        logger.error(f"Error translating answer: {e}")
        translated_answer = answer  # Use original answer if translation fails

    preprocessed_answer = preprocess_text(translated_answer)
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

    # Calculate similarity scores for each criteria
    similarity_scores = {}
    for score, description in translated_criteria.items():
        if isinstance(description, str) and description.strip():
            similarity_scores[score] = calculate_similarity(preprocessed_answer, description)
        else:
            logger.warning(f"Invalid criteria for score {score}: {description}")
            similarity_scores[score] = 0.0

    try:
        best_score = max(similarity_scores, key=similarity_scores.get)
        best_similarity = similarity_scores[best_score]
    except ValueError as e:
        logger.error(f"Error finding best score: {e}. Using default values.")
        best_score = score_given
        best_similarity = 0.0
    
    # Format similarity value to 4 decimal places as a float
    best_similarity_float = float(f"{best_similarity:.4f}")
    
    # Log hasil analisis
    logger.info(f"Hasil Analisis: best_score={best_score}, best_similarity={best_similarity_float}, sentiment={sentiment}")
    logger.info(f"Similarity Scores: {json.dumps({k: float(f'{v:.4f}') for k, v in similarity_scores.items()}, indent=2)}")

    # Return different responses for web form vs API
    if request.is_json:
        response_data = {
            'best_score': best_score,
            'best_similarity': best_similarity_float,
            'sentiment': sentiment,
            'avg_pos': float(f"{avg_pos:.4f}"),
            'avg_neg': float(f"{avg_neg:.4f}"),
            'similarity_scores': {k: float(f"{v:.4f}") for k, v in similarity_scores.items()}
        }
        
        # Log response yang dikirim
        logger.info(f"Response dikirim ke Laravel: {json.dumps(response_data, indent=2)}")
        
        return jsonify(response_data) 
    else:
        return render_template('result.html', answer=answer, score_given=score_given,
                              sentiment=sentiment, avg_pos=avg_pos, avg_neg=avg_neg,
                              best_score=best_score, best_similarity=f"{best_similarity:.4f}")

if __name__ == '__main__':
    logger.info("Aplikasi Flask dimulai")
    app.run(debug=True, host='0.0.0.0')