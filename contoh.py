import flask
import nltk
import spacy
import re
import os  # Tambahkan import os
from functools import lru_cache
from typing import List, Dict, Any, Tuple

# Download NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

# Import NLTK resources after download
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk

class TextAnalyzer:
    def __init__(self):
        # Load spaCy model with caching
        self.nlp = spacy.load("en_core_web_sm")
        
        # Caching for synsets and glosses
        self.synset_cache = {}
        
        # Negative indicators for more robust detection
        self.negative_indicators = [
            'not', 'no', 'never', 'nothing', 'cannot', 'can\'t', 
            'didnt', 'didn\'t', 'unsuccessful', 'failed', 
            'incomplete', 'insufficient', 'lacking'
        ]
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Advanced text preprocessing."""
        # Expand common contractions and shorthands
        expansions = {
            'iam': 'i am',
            'its': 'it is',
            'im': 'i am',
            'cant': 'cannot',
            'dont': 'do not'
        }
        
        # Replace known contractions
        for short, full in expansions.items():
            text = re.sub(r'\b{}\b'.format(short), full, text, flags=re.IGNORECASE)
        
        # Remove extra whitespaces and convert to lowercase
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    def is_negative_response(self, text: str) -> bool:
        """Detect negative or incomplete responses."""
        return any(indicator in text.lower() for indicator in self.negative_indicators)
    
    def calculate_similarity(self, answer_text: str, criteria_text: str) -> Tuple[float, List[Dict], int, Dict]:
        """
        Enhanced similarity calculation with robust matching and scoring
        
        Args:
            answer_text (str): User's response text
            criteria_text (str): Reference criteria text
        
        Returns:
            Tuple containing similarity score, similar words, criteria word count, and match details
        """
        # Preprocess inputs
        answer_text = self.preprocess_text(answer_text)
        criteria_text = self.preprocess_text(criteria_text)
        
        # Extremely negative or irrelevant response
        if self.is_negative_response(answer_text):
            return 0.1, [], 0, {
                'exact_matches': 0,
                'lemma_matches': 0,
                'pos_matches': 0,
                'total_criteria_keywords': 0
            }
        
        # Tokenize with spaCy
        answer_doc = self.nlp(answer_text)
        criteria_doc = self.nlp(criteria_text)
        
        # Advanced similarity scoring function
        def advanced_similarity_scoring(answer_tokens, criteria_tokens):
            total_score = 0
            max_possible_score = len(criteria_tokens)
            
            matching_details = []
            similar_words = []
            
            for c_token in criteria_tokens:
                best_match_score = 0
                best_match = None
                
                for a_token in answer_tokens:
                    # Multiple matching criteria with weighted scoring
                    match_score = (
                        1.0 if a_token.text.lower() == c_token.text.lower() else 0 +  # Exact match
                        0.7 if a_token.lemma_.lower() == c_token.lemma_.lower() else 0 +  # Lemma match
                        0.5 if a_token.pos_ == c_token.pos_ else 0  # POS match
                    )
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = a_token
                
                total_score += best_match_score
                
                # Try to get synset and gloss for similar words
                if best_match:
                    try:
                        wn_pos = {
                            'NOUN': wn.NOUN,
                            'VERB': wn.VERB,
                            'ADJ': wn.ADJ,
                            'ADV': wn.ADV
                        }.get(best_match.pos_)
                        
                        if wn_pos:
                            synset = lesk(answer_text, best_match.text, wn_pos)
                            
                            similar_word_info = {
                                'word': best_match.text,
                                'gloss': synset.definition() if synset else 'No definition available',
                                'negated': any(child.dep_ == 'neg' for child in best_match.children)
                            }
                            
                            similar_words.append(similar_word_info)
                    except Exception:
                        pass
                
                # Store matching details
                matching_details.append({
                    'criteria_word': c_token.text,
                    'matched_word': best_match.text if best_match else 'No match',
                    'match_score': best_match_score
                })
            
            # Normalize score
            normalized_similarity = total_score / max_possible_score if max_possible_score > 0 else 0
            return normalized_similarity, matching_details, similar_words
        
        # Filter out stop words and punctuation
        answer_tokens = [token for token in answer_doc if not token.is_stop and not token.is_punct]
        criteria_tokens = [token for token in criteria_doc if not token.is_stop and not token.is_punct]
        
        # Compute advanced similarity
        precise_similarity, matching_details, similar_words = advanced_similarity_scoring(answer_tokens, criteria_tokens)
        semantic_similarity = answer_doc.similarity(criteria_doc)
        
        # Combined score with weighted averaging
        combined_score = (0.6 * precise_similarity) + (0.4 * semantic_similarity)
        
        # Detailed match metrics
        match_details = {
            'exact_matches': sum(1 for c in criteria_tokens for a in answer_tokens if a.text.lower() == c.text.lower()),
            'lemma_matches': sum(1 for c in criteria_tokens for a in answer_tokens if a.lemma_.lower() == c.lemma_.lower()),
            'pos_matches': sum(1 for c in criteria_tokens for a in answer_tokens if a.pos_ == c.pos_),
            'total_criteria_keywords': len(criteria_tokens)
        }
        
        return combined_score, similar_words, len(criteria_tokens), match_details
    
    def _enhanced_sentiment_analysis(self, text: str) -> Dict:
        """Enhanced sentiment analysis with context-aware processing."""
        doc = self.nlp(text)
        
        # Sentiment tracking
        sentiment_scores = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        word_sentiments = []
        context_sentiment = {
            'positive_contexts': [],
            'negative_contexts': [],
            'neutral_contexts': []
        }
        
        for sent in doc.sents:
            sent_polarity = 0
            
            for token in sent:
                if token.is_stop or token.is_punct:
                    continue
                
                try:
                    # Get WordNet POS
                    wn_pos = {
                        'NOUN': wn.NOUN,
                        'VERB': wn.VERB,
                        'ADJ': wn.ADJ,
                        'ADV': wn.ADV
                    }.get(token.pos_)
                    
                    if wn_pos:
                        synset = lesk(str(sent), token.text, wn_pos)
                        if synset:
                            swn_synset = swn.senti_synset(synset.name())
                            
                            # Detect negation
                            is_negated = any(child.dep_ == 'neg' for child in token.children)
                            
                            # Adjust sentiment scores
                            pos_score = swn_synset.pos_score()
                            neg_score = swn_synset.neg_score()
                            
                            if is_negated:
                                pos_score, neg_score = neg_score, pos_score
                            
                            # Track word-level sentiment
                            word_sentiments.append({
                                'word': token.text,
                                'pos_score': pos_score,
                                'neg_score': neg_score,
                                'is_negated': is_negated
                            })
                            
                            sent_polarity += (pos_score - neg_score)
                            
                except Exception:
                    continue
            
            # Categorize sentence sentiment
            if sent_polarity > 0:
                context_sentiment['positive_contexts'].append(str(sent))
                sentiment_scores['positive'] += 1
            elif sent_polarity < 0:
                context_sentiment['negative_contexts'].append(str(sent))
                sentiment_scores['negative'] += 1
            else:
                context_sentiment['neutral_contexts'].append(str(sent))
                sentiment_scores['neutral'] += 1
        
        # Determine overall sentiment
        total_sentences = sum(sentiment_scores.values())
        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get) if total_sentences > 0 else 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment.capitalize(),
            'sentiment_scores': sentiment_scores,
            'context_sentiment': context_sentiment,
            'word_sentiments': word_sentiments
        }
    
    def _get_word_definitions(self, text: str) -> List[Dict]:
        """
        Retrieve word definitions using NLTK and WordNet
        
        Args:
            text (str): Input text to extract word definitions
        
        Returns:
            List of dictionaries with word definitions
        """
        doc = self.nlp(text)
        word_glosses = []
        
        for token in doc:
            # Skip stop words, punctuation, and very short tokens
            if token.is_stop or token.is_punct or len(token.text) < 2:
                continue
            
            try:
                # Get WordNet POS
                wn_pos = {
                    'NOUN': wn.NOUN,
                    'VERB': wn.VERB,
                    'ADJ': wn.ADJ,
                    'ADV': wn.ADV
                }.get(token.pos_)
                
                if wn_pos:
                    # Use Lesk algorithm to get the most appropriate synset
                    synset = lesk(text, token.text, wn_pos)
                    
                    if synset:
                        # Get definition (gloss)
                        gloss = synset.definition()
                        
                        word_glosses.append({
                            'word': token.text,
                            'pos': token.pos_,
                            'gloss': gloss if gloss else 'No definition available'
                        })
            
            except Exception:
                continue
        
        return word_glosses

class SimilarityAssessmentApp:
    def __init__(self):
        self.app = flask.Flask(__name__)
        self.analyzer = TextAnalyzer()
        self._setup_routes()
        
        # Criteria definitions
        self.criteria = {
            1: "Did not collect any data at all",
            2: "Collected a small portion of the data, incomplete",
            3: "Data has been collected, but there are some deficiencies",
            4: "Data collected is quite good and meets the requirements",
            5: "Data collected is perfect and has been validated"
        }
        
        self.question = "Have you collected the data according to the requirements?"
    
    def _setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            try:
                return flask.render_template('index.html', 
                                             question=self.question, 
                                             criteria=self.criteria)
            except Exception as e:
                return f"<h1>Simple Assessment Tool</h1><p>Question: {self.question}</p><p>Error loading template: {str(e)}</p>"
        
        @self.app.route('/health')
        def health_check():
            return {'status': 'healthy', 'message': 'Application is running'}
        
        @self.app.route('/assess', methods=['POST'])
        def assess():
            try:
                return self._process_assessment()
            except Exception as e:
                return f"<h1>Error Processing Assessment</h1><p>Error: {str(e)}</p><a href='/'>Back to Home</a>"
    
    def _process_assessment(self):
        """Process answer assessment with advanced analysis."""
        # Get form data
        answer = flask.request.form['answer']
        score_given = int(flask.request.form['score'])
        
        # Preprocess answer
        preprocessed_answer = self.analyzer.preprocess_text(answer)
        
        # Enhanced Sentiment Analysis
        sentiment_analysis = self.analyzer._enhanced_sentiment_analysis(preprocessed_answer)
        
        # Similarity Results
        similarity_results = self._compute_similarity_results(preprocessed_answer)
        
        # Best Match Calculation
        best_match_data = self._find_best_match(similarity_results)
        
        # Get word definitions and glosses
        answer_glosses = self.analyzer._get_word_definitions(answer)
        
        # Remove similar_words from best_match_data to avoid duplicate argument
        best_match_data_copy = best_match_data.copy()
        similar_words = best_match_data_copy.pop('similar_words', [])
        
        return flask.render_template('result.html', 
                                    answer=answer, 
                                    score_given=score_given,
                                    sentiment_analysis=sentiment_analysis,
                                    answer_glosses=answer_glosses,
                                    similar_words=similar_words,
                                    **best_match_data_copy,
                                    criteria=self.criteria,
                                    all_similarity_results=similarity_results)
    
    def _compute_similarity_results(self, answer):
        """Compute similarity for all criteria."""
        results = {}
        for score, description in self.criteria.items():
            similarity, similar_words, criteria_word_count, match_details = self.analyzer.calculate_similarity(answer, description)
            
            results[score] = {
                'similarity': similarity,
                'similar_words': similar_words,
                'criteria_word_count': criteria_word_count,
                'match_details': match_details
            }
        
        # Rank-based match percentages
        sorted_scores = sorted(results.items(), key=lambda x: x[1]['similarity'], reverse=True)
        max_similarity = sorted_scores[0][1]['similarity'] if sorted_scores else 0
        
        for score in results:
            results[score]['match_percentage'] = (
                (results[score]['similarity'] / max_similarity) * 100 
                if max_similarity > 0 else 0
            )
        
        return results
    
    def _find_best_match(self, similarity_results):
        """Find best match from similarity results."""
        # Carefully choose the best match
        sorted_scores = sorted(similarity_results.items(), key=lambda x: x[1]['similarity'], reverse=True)
        
        # Choose the best matching score
        best_score = sorted_scores[0][0]
        
        return {
            'best_score': best_score,
            'best_similarity': f"{similarity_results[best_score]['similarity']:.4f}",
            'best_match_percentage': f"{similarity_results[best_score]['match_percentage']:.2f}",
            'similar_words': similarity_results[best_score].get('similar_words', []),
            'match_details': similarity_results[best_score]['match_details']
        }
    
    def run(self, debug=True):
        """Run the Flask application."""
        self.app.run(debug=debug)


similarity_app = SimilarityAssessmentApp()
app = similarity_app.app  

# Main Execution - Updated untuk Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
