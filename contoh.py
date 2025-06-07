import flask
import nltk
import spacy
import re
import os
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import ssl

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
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Failed to download {resource}: {e}")
            if resource == 'punkt_tab':
                try:
                    nltk.download('punkt', quiet=True)
                    print("Downloaded punkt as fallback for punkt_tab")
                except Exception as fallback_error:
                    print(f"Fallback punkt download also failed: {fallback_error}")

download_nltk_resources()

try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    print("All NLTK imports successful")
except ImportError as e:
    print(f"NLTK import error: {e}")
    raise

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
            print(f"Error in simplified_lesk for {word}: {e}")
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
            
            # Balanced sentiment weights
            self.sentiment_weights = {
                'positive_boost': 1.2,
                'negative_penalty': 0.7,
                'neutral_baseline': 1.0
            }
            
            print("Optimized TextAnalyzer initialized successfully")
        except Exception as e:
            print(f"Error initializing TextAnalyzer: {e}")
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
            print(f"Error in preprocess_text: {e}")
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
            print(f"Error in sentiment analysis: {e}")
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
            print(f"Error in SentiWordNet analysis: {e}")
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
                    print(f"Error getting SentiWordNet synset: {e}")
                    return {}
            
            return {}
        except Exception as e:
            print(f"Error getting token sentiment for {token.text}: {e}")
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
            print(f"Error in similarity calculation: {e}")
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
            print(f"Error in token similarity calculation: {e}")
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
            print(f"Error getting word info: {e}")
        
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

class EnhancedSimilarityAssessmentApp:
    def __init__(self):
        self.app = flask.Flask(__name__)
        self.analyzer = OptimizedDynamicTextAnalyzer()
        self._setup_routes()
        
        self.criteria = {
            1: "Did not collect any data at all",
            2: "Collected a small portion of the data, incomplete",
            3: "Data has been collected, but there are some deficiencies",
            4: "Data collected is quite good and meets the requirements",
            5: "Data collected is perfect and has been validated"
        }
        
        self.question = "Have you collected the data according to the requirements?"
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            try:
                return flask.render_template('index.html', 
                                             question=self.question, 
                                             criteria=self.criteria)
            except Exception as e:
                return f"<h1>Enhanced Assessment Tool</h1><p>Question: {self.question}</p><p>Error loading template: {str(e)}</p>"
        
        @self.app.route('/health')
        def health_check():
            return {'status': 'healthy', 'message': 'Optimized application is running'}
        
        @self.app.route('/assess', methods=['POST'])
        def assess():
            try:
                return self._process_assessment()
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return f"<h1>Error Processing Assessment</h1><p>Error: {str(e)}</p><pre>{error_details}</pre><a href='/'>Back to Home</a>"
    
    def _process_assessment(self):
        try:
            answer = flask.request.form['answer']
            score_given = int(flask.request.form['score'])
            
            # Balanced sentiment analysis
            sentiment_analysis = self.analyzer.analyze_balanced_sentiment(answer)
            
            # Transform for template
            transformed_sentiment = self._transform_sentiment(sentiment_analysis)
            
            # Calculate similarities
            similarity_results = self._compute_similarity_results(answer)
            
            # Find best match
            best_match_data = self._find_best_match(similarity_results)
            
            # Get word definitions
            try:
                answer_glosses = self._get_answer_glosses(answer)
            except Exception as e:
                print(f"Error getting answer glosses: {e}")
                answer_glosses = []
            
            similar_words = best_match_data.pop('similar_words', [])
            
            return flask.render_template('result.html', 
                                        answer=answer, 
                                        score_given=score_given,
                                        sentiment_analysis=transformed_sentiment,
                                        answer_glosses=answer_glosses,
                                        similar_words=similar_words,
                                        **best_match_data,
                                        criteria=self.criteria,
                                        all_similarity_results=similarity_results)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in _process_assessment: {e}")
            return f"<h1>Error Processing Assessment</h1><p>Error: {str(e)}</p><pre>{error_details}</pre><a href='/'>Back to Home</a>"
    
    def _compute_similarity_results(self, answer):
        results = {}
        
        for score, description in self.criteria.items():
            similarity, similar_words, criteria_word_count, match_details = (
                self.analyzer.calculate_optimized_similarity(answer, description)
            )
            
            results[score] = {
                'similarity': similarity,
                'similar_words': similar_words,
                'criteria_word_count': criteria_word_count,
                'match_details': match_details
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
    
    def _find_best_match(self, similarity_results):
        sorted_scores = sorted(similarity_results.items(), 
                             key=lambda x: x[1]['similarity'], reverse=True)
        
        best_score = sorted_scores[0][0]
        
        return {
            'best_score': best_score,
            'best_similarity': f"{similarity_results[best_score]['similarity']:.4f}",
            'best_match_percentage': f"{similarity_results[best_score]['match_percentage']:.2f}",
            'similar_words': similarity_results[best_score].get('similar_words', []),
            'match_details': similarity_results[best_score]['match_details']
        }
    
    def _transform_sentiment(self, sentiment_analysis: Dict) -> Dict:
        vader_data = sentiment_analysis.get('vader_analysis', {}).get('overall_scores', {})
        
        return {
            'overall_sentiment': sentiment_analysis.get('sentiment_label', 'Neutral'),
            'sentiment_scores': {
                'positive': int(vader_data.get('pos', 0) * 10),
                'negative': int(vader_data.get('neg', 0) * 10),
                'neutral': int(vader_data.get('neu', 0) * 10)
            },
            'context_sentiment': {
                'positive_contexts': [],
                'negative_contexts': [],
                'neutral_contexts': []
            },
            'word_sentiments': sentiment_analysis.get('sentiwordnet_analysis', {}).get('word_analyses', []),
            'confidence': sentiment_analysis.get('confidence', 0),
            'overall_polarity': sentiment_analysis.get('overall_polarity', 0)
        }
    
    def _get_answer_glosses(self, answer: str) -> List[Dict]:
        doc = self.analyzer.nlp(answer)
        glosses = []
        
        for token in doc:
            if self.analyzer._is_meaningful_token(token):
                word_info = self.analyzer._get_word_info(token, answer)
                if word_info and word_info['gloss'] != 'No definition available':
                    glosses.append({
                        'word': word_info['word'],
                        'pos': word_info['pos'],
                        'gloss': word_info['gloss']
                    })
        
        return glosses
    
    def run(self, debug=True):
        self.app.run(debug=debug)

# Initialize app
enhanced_similarity_app = EnhancedSimilarityAssessmentApp()
app = enhanced_similarity_app.app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)