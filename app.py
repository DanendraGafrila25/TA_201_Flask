from flask import Flask, render_template, request, jsonify
import nltk
import spacy
import re
import json
import os
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import ssl
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from translate import Translator

# Initialize translator
translator = Translator(to_lang="en", from_lang="id")

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Enhanced NLTK resource downloads with error handling
def download_nltk_resources():
    """Download NLTK resources with fallback handling"""
    resources = [
        'punkt_tab',
        'punkt',
        'wordnet', 
        'sentiwordnet', 
        'averaged_perceptron_tagger', 
        'stopwords', 
        'vader_lexicon'
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            if resource == 'punkt_tab':
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as fallback_error:
                    pass

download_nltk_resources()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class OptimizedLeskAlgorithm:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.cache = {}
    
    @lru_cache(maxsize=2000)
    def simplified_lesk(self, word: str, context: str, pos: str = None) -> Any:
        """Optimized Lesk algorithm with enhanced context matching"""
        cache_key = f"{word}_{pos}_{hash(context[:100])}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get context words (excluding stop words)
            context_words = set([w.lower() for w in word_tokenize(context.lower()) 
                               if w.isalpha() and w not in self.stop_words])
            
            # Get synsets for the word
            if pos:
                synsets = wn.synsets(word.lower(), pos=pos)
            else:
                synsets = wn.synsets(word.lower())
            
            if not synsets:
                return None
            
            best_synset = None
            max_overlap = -1
            
            for synset in synsets:
                # Get definition and examples
                definition = synset.definition().lower()
                examples = ' '.join(synset.examples()).lower()
                
                # Get hypernyms and hyponyms for broader context
                related_words = set()
                for hypernym in synset.hypernyms():
                    related_words.update([w.lower() for w in word_tokenize(hypernym.definition().lower())])
                for hyponym in synset.hyponyms():
                    related_words.update([w.lower() for w in word_tokenize(hyponym.definition().lower())])
                
                # Combine all synset text
                synset_text = f"{definition} {examples}"
                synset_words = set([w for w in word_tokenize(synset_text) 
                                  if w.isalpha() and w not in self.stop_words])
                synset_words.update(related_words)
                
                # Calculate overlap with enhanced scoring
                overlap = len(context_words.intersection(synset_words))
                
                # Bonus for exact word matches in definition
                exact_matches = sum(1 for w in context_words if w in definition)
                overlap += exact_matches * 2
                
                # Bonus for lemma matches
                lemma_matches = sum(1 for w in context_words 
                                  if any(lemma.name().lower() == w for lemma in synset.lemmas()))
                overlap += lemma_matches * 1.5
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_synset = synset
            
            self.cache[cache_key] = best_synset
            return best_synset
            
        except Exception as e:
            return synsets[0] if synsets else None

class OptimizedDynamicTextAnalyzer:
    def __init__(self):
        """Initialize analyzer with optimized components."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.lemmatizer = WordNetLemmatizer()
            self.sia = SentimentIntensityAnalyzer()
            self.lesk = OptimizedLeskAlgorithm()
            
            self.stop_words = set(stopwords.words('english'))
            
            # Optimized caching
            self.cache = {
                'sentiment': {},
                'similarity': {},
                'synsets': {},
                'token_analysis': {}
            }
            
        except Exception as e:
            raise
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Optimized text preprocessing"""
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text.lower())
            processed_tokens = []
            
            for token in tokens:
                if token.isalpha() and len(token) > 1:
                    if token not in self.stop_words:
                        lemma = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)
            
            return ' '.join(processed_tokens)
        except Exception as e:
            words = text.lower().split()
            return ' '.join([word for word in words if word.isalpha() and len(word) > 1])
    
    def analyze_balanced_sentiment(self, text: str) -> Dict[str, Any]:
        """Balanced sentiment analysis without negative bias"""
        try:
            # VADER analysis
            vader_scores = self.sia.polarity_scores(text)
            
            # SentiWordNet analysis with optimized Lesk
            swn_analysis = self._analyze_with_optimized_sentiwordnet(text)
            
            # Determine sentiment based on multiple factors
            is_positive = self._determine_positivity(vader_scores, swn_analysis)
            is_negative = self._determine_negativity(vader_scores, swn_analysis)
            
            # Calculate balanced polarity
            polarity = self._calculate_balanced_polarity(vader_scores, swn_analysis)
            
            return {
                'vader_analysis': {'overall_scores': vader_scores},
                'sentiwordnet_analysis': swn_analysis,
                'is_positive': is_positive,
                'is_negative': is_negative,
                'overall_polarity': polarity,
                'confidence': abs(polarity),
                'sentiment_label': self._get_sentiment_label(polarity)
            }
        except Exception as e:
            return self._get_default_sentiment()
    
    def _analyze_with_optimized_sentiwordnet(self, text: str) -> Dict[str, Any]:
        """Optimized SentiWordNet analysis using enhanced Lesk"""
        try:
            doc = self.nlp(text)
            sentiment_scores = {'positive': 0, 'negative': 0, 'objective': 0}
            word_analyses = []
            
            for token in doc:
                if self._is_meaningful_token(token):
                    token_sentiment = self._get_optimized_token_sentiment(token, text)
                    
                    if token_sentiment:
                        sentiment_scores['positive'] += token_sentiment['pos_score']
                        sentiment_scores['negative'] += token_sentiment['neg_score']
                        sentiment_scores['objective'] += token_sentiment['obj_score']
                        
                        word_analyses.append({
                            'word': token.text,
                            'pos_score': token_sentiment['pos_score'],
                            'neg_score': token_sentiment['neg_score'],
                            'obj_score': token_sentiment['obj_score']
                        })
            
            # Calculate normalized polarity
            total_sentiment = sentiment_scores['positive'] + sentiment_scores['negative']
            if total_sentiment > 0:
                polarity = (sentiment_scores['positive'] - sentiment_scores['negative']) / total_sentiment
            else:
                polarity = 0
            
            return {
                'scores': sentiment_scores,
                'polarity': polarity,
                'word_analyses': word_analyses
            }
        except Exception as e:
            return {
                'scores': {'positive': 0, 'negative': 0, 'objective': 0},
                'polarity': 0,
                'word_analyses': []
            }
    
    def _get_optimized_token_sentiment(self, token, context: str) -> Dict:
        """Get token sentiment using optimized Lesk algorithm"""
        try:
            cache_key = f"{token.text}_{token.pos_}"
            
            if cache_key in self.cache['token_analysis']:
                return self.cache['token_analysis'][cache_key]
            
            pos_mapping = {
                'NOUN': wn.NOUN, 'PROPN': wn.NOUN,
                'VERB': wn.VERB, 'AUX': wn.VERB,
                'ADJ': wn.ADJ, 'ADV': wn.ADV
            }
            
            wn_pos = pos_mapping.get(token.pos_)
            if not wn_pos:
                return {}
            
            # Use optimized Lesk algorithm
            synset = self.lesk.simplified_lesk(token.text, context, wn_pos)
            
            if synset:
                try:
                    swn_synset = swn.senti_synset(synset.name())
                    sentiment_data = {
                        'pos_score': swn_synset.pos_score(),
                        'neg_score': swn_synset.neg_score(),
                        'obj_score': swn_synset.obj_score()
                    }
                    
                    self.cache['token_analysis'][cache_key] = sentiment_data
                    return sentiment_data
                except Exception as e:
                    return {}
            
            return {}
        except Exception as e:
            return {}
    
    def _determine_positivity(self, vader_scores: Dict, swn_analysis: Dict) -> bool:
        """Determine if sentiment is positive"""
        vader_positive = vader_scores['compound'] > 0.1
        swn_positive = swn_analysis['polarity'] > 0.1
        return vader_positive or swn_positive
    
    def _determine_negativity(self, vader_scores: Dict, swn_analysis: Dict) -> bool:
        """Determine if sentiment is negative"""
        vader_negative = vader_scores['compound'] < -0.1
        swn_negative = swn_analysis['polarity'] < -0.1
        return vader_negative and swn_negative
    
    def _calculate_balanced_polarity(self, vader_scores: Dict, swn_analysis: Dict) -> float:
        """Calculate balanced polarity without bias"""
        vader_polarity = vader_scores['compound']
        swn_polarity = swn_analysis['polarity']
        
        # Weighted average
        combined_polarity = (vader_polarity * 0.6) + (swn_polarity * 0.4)
        return combined_polarity
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Get sentiment label based on polarity"""
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    def calculate_optimized_similarity(self, answer_text: str, criteria_text: str) -> Tuple[float, List[Dict], int, Dict]:
        """Optimized similarity calculation"""
        try:
            # Sentiment analysis
            sentiment_analysis = self.analyze_balanced_sentiment(answer_text)
            
            # Preprocess texts
            answer_processed = self.preprocess_text(answer_text)
            criteria_processed = self.preprocess_text(criteria_text)
            
            # Calculate similarities
            answer_doc = self.nlp(answer_processed)
            criteria_doc = self.nlp(criteria_processed)
            base_similarity = answer_doc.similarity(criteria_doc)
            
            # Token-level similarity
            token_similarity, similar_words, match_details = self._calculate_token_similarity(
                answer_doc, criteria_doc, answer_text
            )
            
            # Semantic similarity boost for positive answers
            semantic_boost = self._calculate_semantic_boost(sentiment_analysis, criteria_text)
            
            # Combine similarities
            final_similarity = self._combine_similarities(
                base_similarity, token_similarity, semantic_boost
            )
            
            return (final_similarity, similar_words, len([t for t in criteria_doc if not t.is_stop]), match_details)
        except Exception as e:
            return (0.0, [], 0, {})
    
    def _calculate_semantic_boost(self, sentiment_analysis: Dict, criteria_text: str) -> float:
        """Calculate semantic boost based on sentiment and criteria context"""
        polarity = sentiment_analysis['overall_polarity']
        
        # Analyze criteria to understand expected sentiment
        criteria_doc = self.nlp(criteria_text)
        criteria_keywords = {'perfect', 'good', 'complete', 'validated', 'meets', 'requirements'}
        
        positive_criteria = any(token.text.lower() in criteria_keywords for token in criteria_doc)
        
        if positive_criteria and polarity > 0.05:
            # Positive answer for positive criteria
            return min(0.2, polarity * 0.5)
        elif positive_criteria and polarity < -0.05:
            # Negative answer for positive criteria
            return max(-0.1, polarity * 0.3)
        else:
            return 0.0
    
    def _combine_similarities(self, base_sim: float, token_sim: float, semantic_boost: float) -> float:
        """Combine similarities with balanced weighting"""
        # Balanced combination
        combined = (0.3 * base_sim) + (0.7 * token_sim) + semantic_boost
        return max(0.0, min(1.0, combined))
    
    def _calculate_token_similarity(self, answer_doc, criteria_doc, original_answer: str) -> Tuple[float, List[Dict], Dict]:
        """Calculate token-level similarity"""
        try:
            answer_tokens = [t for t in answer_doc if self._is_meaningful_token(t)]
            criteria_tokens = [t for t in criteria_doc if self._is_meaningful_token(t)]
            
            if not criteria_tokens:
                return 0.0, [], {}
            
            total_score = 0
            similar_words = []
            match_counts = {'exact': 0, 'lemma': 0, 'semantic': 0}
            
            for c_token in criteria_tokens:
                best_score = 0
                best_match = None
                
                for a_token in answer_tokens:
                    # Exact match
                    if c_token.text.lower() == a_token.text.lower():
                        score = 1.0
                        match_counts['exact'] += 1
                    # Lemma match
                    elif c_token.lemma_.lower() == a_token.lemma_.lower():
                        score = 0.9
                        match_counts['lemma'] += 1
                    # Semantic similarity
                    else:
                        score = self._calculate_wordnet_similarity(c_token.text, a_token.text)
                        if score > 0.3:
                            match_counts['semantic'] += 1
                    
                    if score > best_score:
                        best_score = score
                        best_match = a_token
                
                total_score += best_score
                
                if best_match and best_score > 0.3:
                    word_info = self._get_word_info(best_match, original_answer)
                    if word_info:
                        similar_words.append(word_info)
            
            avg_similarity = total_score / len(criteria_tokens)
            
            match_details = {
                'exact_matches': match_counts['exact'],
                'lemma_matches': match_counts['lemma'],
                'semantic_matches': match_counts['semantic'],
                'total_criteria_keywords': len(criteria_tokens)
            }
            
            return avg_similarity, similar_words, match_details
        except Exception as e:
            return 0.0, [], {}
    
    def _calculate_wordnet_similarity(self, word1: str, word2: str) -> float:
        """Calculate WordNet-based similarity"""
        try:
            synsets1 = wn.synsets(word1.lower())
            synsets2 = wn.synsets(word2.lower())
            
            if not synsets1 or not synsets2:
                return 0.0
            
            max_sim = 0
            for s1 in synsets1[:2]:
                for s2 in synsets2[:2]:
                    try:
                        sim = s1.path_similarity(s2)
                        if sim and sim > max_sim:
                            max_sim = sim
                    except:
                        continue
            
            return max_sim * 0.7
        except Exception as e:
            return 0.0
    
    def _is_meaningful_token(self, token) -> bool:
        """Check if token is meaningful for analysis"""
        try:
            return (not token.is_stop and not token.is_punct and 
                    not token.is_space and len(token.text) > 1 and token.text.isalpha())
        except:
            return len(token.text) > 1 and token.text.isalpha()
    
    def _get_word_info(self, token, context: str) -> Dict:
        """Get word information with optimized Lesk"""
        try:
            pos_mapping = {
                'NOUN': wn.NOUN, 'PROPN': wn.NOUN,
                'VERB': wn.VERB, 'AUX': wn.VERB,
                'ADJ': wn.ADJ, 'ADV': wn.ADV
            }
            
            wn_pos = pos_mapping.get(token.pos_)
            if wn_pos:
                synset = self.lesk.simplified_lesk(token.text, context, wn_pos)
                if synset:
                    return {
                        'word': token.text,
                        'gloss': synset.definition(),
                        'pos': token.pos_
                    }
        except Exception as e:
            pass
        
        return {
            'word': token.text,
            'gloss': 'No definition available',
            'pos': getattr(token, 'pos_', 'Unknown')
        }
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default sentiment structure"""
        return {
            'vader_analysis': {'overall_scores': {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}},
            'sentiwordnet_analysis': {'polarity': 0, 'word_analyses': []},
            'is_positive': False,
            'is_negative': False,
            'overall_polarity': 0,
            'confidence': 0,
            'sentiment_label': 'Neutral'
        }

app = Flask(__name__)

# Initialize the enhanced analyzer
analyzer = OptimizedDynamicTextAnalyzer()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\biam\b', 'i am', text, flags=re.IGNORECASE)
    text = re.sub(r'\bits\b', 'it is', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Translation function
def translate_text(text):
    try:
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        return text  # Return original text if translation fails

# Similarity calculation using the prototype's methodology
def compute_similarity_results(answer, criteria_dict):
    """Compute similarity results using prototype's methodology"""
    results = {}
    
    for score, description in criteria_dict.items():
        if isinstance(description, str) and description.strip():
            similarity, similar_words, criteria_word_count, match_details = (
                analyzer.calculate_optimized_similarity(answer, description)
            )
            
            results[score] = {
                'similarity': similarity,
                'similar_words': similar_words,
                'criteria_word_count': criteria_word_count,
                'match_details': match_details
            }
        else:
            results[score] = {
                'similarity': 0.0,
                'similar_words': [],
                'criteria_word_count': 0,
                'match_details': {}
            }
    
    # Calculate match percentages
    similarities = [results[score]['similarity'] for score in results]
    max_similarity = max(similarities) if similarities else 0
    
    for score in results:
        if max_similarity > 0:
            results[score]['match_percentage'] = (results[score]['similarity'] / max_similarity) * 100
        else:
            results[score]['match_percentage'] = 0
    
    return results

def find_best_match(similarity_results):
    """Find best match using prototype's methodology"""
    sorted_scores = sorted(similarity_results.items(), 
                         key=lambda x: x[1]['similarity'], reverse=True)
    
    best_score = sorted_scores[0][0]
    
    return {
        'best_score': best_score,
        'best_similarity': similarity_results[best_score]['similarity'],
        'best_match_percentage': similarity_results[best_score]['match_percentage'],
        'similar_words': similarity_results[best_score].get('similar_words', []),
        'match_details': similarity_results[best_score]['match_details']
    }

# Default criteria for fallback
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
            return None
        
        with open(rubrik_file, 'r', encoding='utf-8') as f:
            rubrik_data = json.load(f)
        
        # Match based on question_id which corresponds to assessment_id in Rubrik.json
        for assessment_id, assessment_data in rubrik_data.items():
            if assessment_id == question_id:
                return assessment_data
        
        return None
    except Exception as e:
        return None

@app.route('/import-criteria', methods=['POST'])
def import_criteria():
    if request.is_json:
        data = request.json
        
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
            
            return jsonify({
                'success': True,
                'assessment_id': assessment_id,
                'question': question,
                'translated_criteria': translated_criteria,
                'message': f"Data berhasil disimpan ke dalam {rubrik_file}"
            })
            
        except Exception as e:
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
    # Check if the request is from a API call or web form
    if request.is_json:
        # API request from Laravel
        data = request.json
        
        # Extract the data
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
        else:
            # Use default criteria if no match found
            translated_criteria = default_criteria
        
        # Ensure we have criteria for all scores 1-5
        for i in range(1, 6):
            if i not in translated_criteria:
                translated_criteria[i] = default_criteria[i]
    else:
        # Web form request
        answer = request.form['answer']
        score_given = int(request.form['score'])
        translated_criteria = default_criteria

    # Translate the answer to English
    try:
        translated_answer = translate_text(answer)
    except Exception as e:
        translated_answer = answer  # Use original answer if translation fails

    preprocessed_answer = preprocess_text(translated_answer)
    
    # Enhanced sentiment analysis using the optimized analyzer
    sentiment_analysis = analyzer.analyze_balanced_sentiment(preprocessed_answer)
    
    # Extract sentiment info for compatibility with original return format
    avg_pos = sentiment_analysis['sentiwordnet_analysis']['scores']['positive']
    avg_neg = sentiment_analysis['sentiwordnet_analysis']['scores']['negative']
    sentiment = sentiment_analysis['sentiment_label']

    # Calculate similarities using prototype's methodology
    similarity_results = compute_similarity_results(preprocessed_answer, translated_criteria)
    
    # Find best match using prototype's methodology
    best_match_data = find_best_match(similarity_results)
    
    best_score = best_match_data['best_score']
    best_similarity = best_match_data['best_similarity']
    best_match_percentage = best_match_data['best_match_percentage']
    
    # Format similarity value to 4 decimal places as a float
    best_similarity_float = float(f"{best_similarity:.4f}")
    
    # Create similarity scores dict for compatibility
    similarity_scores = {k: v['similarity'] for k, v in similarity_results.items()}
    
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
        
        return jsonify(response_data) 
    else:
        return render_template('result.html', answer=answer, score_given=score_given,
                              sentiment=sentiment, avg_pos=avg_pos, avg_neg=avg_neg,
                              best_score=best_score, best_similarity=f"{best_similarity:.4f}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')