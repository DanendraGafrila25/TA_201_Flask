import flask
import nltk
import spacy
import re
import os
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Tuple
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
        'punkt_tab',  # Add the missing resource
        'punkt',      # Fallback for punkt_tab
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
            # For punkt_tab, try punkt as fallback
            if resource == 'punkt_tab':
                try:
                    nltk.download('punkt', quiet=True)
                    print("Downloaded punkt as fallback for punkt_tab")
                except Exception as fallback_error:
                    print(f"Fallback punkt download also failed: {fallback_error}")

# Download resources
download_nltk_resources()

# Import NLTK resources with error handling
try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import stopwords
    from nltk.wsd import lesk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    print("All NLTK imports successful")
except ImportError as e:
    print(f"NLTK import error: {e}")
    # You might want to implement fallback tokenization here
    raise

class OptimizedDynamicTextAnalyzer:
    def __init__(self):
        """Initialize analyzer with enhanced dynamic components."""
        try:
            # Core NLP components
            self.nlp = spacy.load("en_core_web_sm")
            self.lemmatizer = WordNetLemmatizer()
            self.sia = SentimentIntensityAnalyzer()
            
            # Dynamic language resources
            self.stop_words = set(stopwords.words('english'))
            
            # Performance caching
            self.cache = {
                'sentiment': {},
                'similarity': {},
                'synsets': {},
                'negation': {}
            }
            
            # Dynamic sentiment weights - more aggressive negative detection
            self.sentiment_weights = {
                'negative_multiplier': 2.5,  # Amplify negative signals
                'positive_dampener': 0.8,    # Reduce positive signals when negative present
                'negation_strength': 3.0     # Strong negation impact
            }
            print("TextAnalyzer initialized successfully")
        except Exception as e:
            print(f"Error initializing TextAnalyzer: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with negation preservation."""
        if not text:
            return ""
        
        try:
            # Preserve negation words during preprocessing
            negation_markers = {'not', 'no', 'never', 'none', 'nothing', 'neither', 
                               'nowhere', 'nobody', 'cannot', 'cant', 'wont', 'dont', 
                               'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent',
                               'hadnt', 'shouldnt', 'wouldnt', 'couldnt', 'mustnt'}
            
            tokens = word_tokenize(text.lower())
            processed_tokens = []
            
            for token in tokens:
                if token.isalpha():
                    if token in negation_markers:
                        # Preserve negation words
                        processed_tokens.append(token)
                    elif token not in self.stop_words:
                        lemma = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)
            
            return ' '.join(processed_tokens)
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            # Fallback to simple preprocessing
            words = text.lower().split()
            return ' '.join([word for word in words if word.isalpha() and len(word) > 1])
    
    def analyze_comprehensive_sentiment(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with stronger negative detection."""
        try:
            # Multi-level sentiment analysis
            vader_analysis = self._analyze_with_vader(text)
            sentiwordnet_analysis = self._analyze_with_sentiwordnet_enhanced(text)
            negation_analysis = self._analyze_negation_patterns(text)
            
            # Enhanced negative detection
            negative_indicators = self._detect_negative_indicators(text)
            
            # Combine all analyses with weighted approach
            comprehensive_result = self._combine_sentiment_analyses(
                vader_analysis, sentiwordnet_analysis, negation_analysis, negative_indicators
            )
            
            return comprehensive_result
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return basic sentiment as fallback
            try:
                scores = self.sia.polarity_scores(text)
                return {
                    'vader_analysis': {'overall_scores': scores},
                    'sentiwordnet_analysis': {'polarity': scores['compound'], 'word_analyses': []},
                    'negation_analysis': {'patterns': [], 'negation_count': 0},
                    'negative_indicators': {'categories': {}, 'total_negative_words': 0, 'negative_strength': 0},
                    'is_negative': scores['compound'] < -0.1,
                    'overall_polarity': scores['compound'],
                    'confidence': abs(scores['compound']),
                    'negative_strength': max(0, -scores['compound'])
                }
            except Exception as fallback_error:
                print(f"Fallback sentiment analysis also failed: {fallback_error}")
                return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default sentiment structure when all else fails"""
        return {
            'vader_analysis': {'overall_scores': {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}},
            'sentiwordnet_analysis': {'polarity': 0, 'word_analyses': []},
            'negation_analysis': {'patterns': [], 'negation_count': 0},
            'negative_indicators': {'categories': {}, 'total_negative_words': 0, 'negative_strength': 0},
            'is_negative': False,
            'overall_polarity': 0,
            'confidence': 0,
            'negative_strength': 0
        }
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Enhanced VADER analysis with context awareness."""
        try:
            scores = self.sia.polarity_scores(text)
            
            # Analyze sentence-level sentiment for more granular understanding
            sentences = sent_tokenize(text)
            sentence_sentiments = []
            
            for sentence in sentences:
                sent_scores = self.sia.polarity_scores(sentence)
                sentence_sentiments.append({
                    'sentence': sentence,
                    'scores': sent_scores,
                    'dominant': self._get_dominant_sentiment(sent_scores)
                })
            
            return {
                'overall_scores': scores,
                'sentence_sentiments': sentence_sentiments,
                'negative_sentence_count': sum(1 for s in sentence_sentiments if s['dominant'] == 'negative'),
                'total_sentences': len(sentence_sentiments)
            }
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return {
                'overall_scores': {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1},
                'sentence_sentiments': [],
                'negative_sentence_count': 0,
                'total_sentences': 0
            }
    
    def _analyze_with_sentiwordnet_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced SentiWordNet analysis with better context handling."""
        try:
            doc = self.nlp(text)
            sentiment_accumulator = defaultdict(float)
            word_analyses = []
            
            for token in doc:
                if self._is_meaningful_token(token):
                    token_analysis = self._get_enhanced_token_sentiment(token, text)
                    
                    if token_analysis:
                        # Handle negation more aggressively
                        negation_context = self._get_negation_context(token)
                        
                        pos_score = token_analysis['pos_score']
                        neg_score = token_analysis['neg_score']
                        obj_score = token_analysis['obj_score']
                        
                        # Apply negation with stronger impact
                        if negation_context['is_negated']:
                            negation_strength = self.sentiment_weights['negation_strength']
                            # Flip and amplify sentiment
                            original_pos = pos_score
                            pos_score = neg_score * negation_strength
                            neg_score = original_pos * negation_strength
                        
                        sentiment_accumulator['positive'] += pos_score
                        sentiment_accumulator['negative'] += neg_score
                        sentiment_accumulator['objective'] += obj_score
                        
                        word_analyses.append({
                            'word': token.text,
                            'pos_score': pos_score,
                            'neg_score': neg_score,
                            'obj_score': obj_score,
                            'negation_context': negation_context,
                            'definition': token_analysis.get('definition', '')
                        })
            
            # Calculate enhanced polarity
            total_sentiment = sentiment_accumulator['positive'] + sentiment_accumulator['negative']
            if total_sentiment > 0:
                polarity = (sentiment_accumulator['positive'] - sentiment_accumulator['negative']) / total_sentiment
            else:
                polarity = 0
            
            return {
                'scores': dict(sentiment_accumulator),
                'polarity': polarity,
                'word_analyses': word_analyses,
                'negative_word_count': sum(1 for w in word_analyses if w['neg_score'] > w['pos_score'])
            }
        except Exception as e:
            print(f"Error in SentiWordNet analysis: {e}")
            return {
                'scores': {'positive': 0, 'negative': 0, 'objective': 0},
                'polarity': 0,
                'word_analyses': [],
                'negative_word_count': 0
            }
    
    def _analyze_negation_patterns(self, text: str) -> Dict[str, Any]:
        """Advanced negation pattern analysis using spaCy."""
        try:
            doc = self.nlp(text)
            negation_patterns = []
            
            for token in doc:
                if token.dep_ == 'neg' or token.text.lower() in ['not', 'no', 'never', 'none']:
                    # Find what this negation affects
                    affected_tokens = self._find_negation_scope(token)
                    
                    negation_patterns.append({
                        'negation_word': token.text,
                        'position': token.i,
                        'affected_tokens': [t.text for t in affected_tokens],
                        'scope_strength': len(affected_tokens)
                    })
            
            return {
                'patterns': negation_patterns,
                'negation_count': len(negation_patterns),
                'has_strong_negation': any(p['scope_strength'] > 2 for p in negation_patterns)
            }
        except Exception as e:
            print(f"Error in negation analysis: {e}")
            return {
                'patterns': [],
                'negation_count': 0,
                'has_strong_negation': False
            }
    
    def _detect_negative_indicators(self, text: str) -> Dict[str, Any]:
        """Detect various negative indicators using NLTK and spaCy."""
        try:
            # Lexical negative indicators
            negative_words = {
                'quality': ['bad', 'poor', 'terrible', 'awful', 'horrible', 'inadequate', 'insufficient'],
                'completion': ['incomplete', 'missing', 'absent', 'lacking', 'deficient'],
                'negation': ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody'],
                'failure': ['failed', 'unsuccessful', 'wrong', 'error', 'mistake', 'problem']
            }
            
            text_lower = text.lower()
            detected_categories = defaultdict(list)
            
            for category, words in negative_words.items():
                for word in words:
                    if word in text_lower:
                        detected_categories[category].append(word)
            
            # Calculate negative indicator strength
            total_negative_words = sum(len(words) for words in detected_categories.values())
            
            return {
                'categories': dict(detected_categories),
                'total_negative_words': total_negative_words,
                'negative_strength': min(1.0, total_negative_words / 5.0)  # Normalize to 0-1
            }
        except Exception as e:
            print(f"Error in negative indicator detection: {e}")
            return {
                'categories': {},
                'total_negative_words': 0,
                'negative_strength': 0
            }
    
    def _combine_sentiment_analyses(self, vader_analysis: Dict, sentiwordnet_analysis: Dict, 
                                  negation_analysis: Dict, negative_indicators: Dict) -> Dict[str, Any]:
        """Combine all sentiment analyses with enhanced negative detection."""
        try:
            # Extract key metrics
            vader_compound = vader_analysis['overall_scores']['compound']
            vader_neg = vader_analysis['overall_scores']['neg']
            swn_polarity = sentiwordnet_analysis['polarity']
            negation_count = negation_analysis['negation_count']
            negative_strength = negative_indicators['negative_strength']
            
            # Enhanced negative detection logic
            is_negative = self._determine_enhanced_negativity(
                vader_compound, vader_neg, swn_polarity, negation_count, negative_strength
            )
            
            # Calculate comprehensive polarity with negative bias
            if is_negative:
                # Apply stronger negative weighting
                polarity = (vader_compound * 0.4 + swn_polarity * 0.3 - negative_strength * 0.3)
                # Ensure negative answers get negative polarity
                polarity = min(polarity, -0.1)
            else:
                polarity = (vader_compound * 0.6 + swn_polarity * 0.4)
            
            # Calculate confidence with negative bias
            confidence = max(
                abs(vader_compound),
                abs(swn_polarity),
                negative_strength
            )
            
            return {
                'vader_analysis': vader_analysis,
                'sentiwordnet_analysis': sentiwordnet_analysis,
                'negation_analysis': negation_analysis,
                'negative_indicators': negative_indicators,
                'is_negative': is_negative,
                'overall_polarity': polarity,
                'confidence': confidence,
                'negative_strength': negative_strength
            }
        except Exception as e:
            print(f"Error combining sentiment analyses: {e}")
            return self._get_default_sentiment()
    
    def _determine_enhanced_negativity(self, vader_compound: float, vader_neg: float, 
                                     swn_polarity: float, negation_count: int, 
                                     negative_strength: float) -> bool:
        """Enhanced negative determination with multiple criteria."""
        try:
            # Multiple negative indicators
            criteria = [
                vader_compound < -0.1,  # VADER suggests negative
                vader_neg > 0.3,        # High VADER negative score
                swn_polarity < -0.1,    # SentiWordNet suggests negative
                negation_count > 0,     # Presence of negation
                negative_strength > 0.2  # Strong negative indicators
            ]
            
            # If any 2 or more criteria are met, consider negative
            negative_votes = sum(criteria)
            
            # Special case: strong negative indicators override other signals
            if negative_strength > 0.4 or negation_count > 1:
                return True
            
            return negative_votes >= 2
        except Exception as e:
            print(f"Error determining negativity: {e}")
            return False
    
    def calculate_enhanced_similarity(self, answer_text: str, criteria_text: str) -> Tuple[float, List[Dict], int, Dict]:
        """Enhanced similarity calculation with sentiment-aware scoring."""
        try:
            # Comprehensive sentiment analysis
            sentiment_analysis = self.analyze_comprehensive_sentiment(answer_text)
            
            # Preprocess texts
            answer_processed = self.preprocess_text(answer_text)
            criteria_processed = self.preprocess_text(criteria_text)
            
            # Calculate base similarity
            answer_doc = self.nlp(answer_processed)
            criteria_doc = self.nlp(criteria_processed)
            base_similarity = answer_doc.similarity(criteria_doc)
            
            # Enhanced token matching
            token_similarity, similar_words, match_details = self._calculate_enhanced_token_similarity(
                answer_doc, criteria_doc, answer_text, sentiment_analysis
            )
            
            # Apply sentiment-based adjustments with stronger negative penalty
            sentiment_adjustment = self._calculate_enhanced_sentiment_adjustment(
                sentiment_analysis, criteria_text
            )
            
            # Combine similarities with sentiment awareness
            final_similarity = self._combine_enhanced_similarities(
                base_similarity, token_similarity, sentiment_adjustment
            )
            
            return (final_similarity, similar_words, len([t for t in criteria_doc if not t.is_stop]), match_details)
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            # Return fallback similarity
            return (0.0, [], 0, {})
    
    def _calculate_enhanced_sentiment_adjustment(self, sentiment_analysis: Dict, criteria_text: str) -> float:
        """Calculate enhanced sentiment adjustment based on criteria context."""
        try:
            is_negative = sentiment_analysis['is_negative']
            negative_strength = sentiment_analysis['negative_strength']
            polarity = sentiment_analysis['overall_polarity']
            
            # Analyze criteria sentiment to understand expected tone
            criteria_sentiment = self.analyze_comprehensive_sentiment(criteria_text)
            criteria_is_positive = criteria_sentiment['overall_polarity'] > 0.1
            
            if is_negative:
                if criteria_is_positive:
                    # Negative answer for positive criteria - strong penalty
                    penalty = 0.3 + (negative_strength * 0.4)
                    return -min(0.7, penalty)
                else:
                    # Negative answer for negative criteria - moderate penalty
                    penalty = 0.1 + (negative_strength * 0.2)
                    return -min(0.3, penalty)
            else:
                # Positive/neutral answer - small bonus
                return min(0.1, abs(polarity) * 0.1)
        except Exception as e:
            print(f"Error in sentiment adjustment: {e}")
            return 0.0
    
    def _combine_enhanced_similarities(self, base_sim: float, token_sim: float, 
                                     sentiment_adj: float) -> float:
        """Enhanced similarity combination with sentiment weighting."""
        try:
            # Dynamic weighting based on sentiment adjustment
            if sentiment_adj < -0.2:  # Strong negative adjustment
                # Prioritize sentiment over similarity for negative cases
                combined = (0.2 * base_sim) + (0.3 * token_sim) + (0.5 * (1 + sentiment_adj))
            else:
                # Normal weighting
                combined = (0.4 * base_sim) + (0.6 * token_sim)
            
            # Apply sentiment adjustment
            adjusted = combined + sentiment_adj
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, adjusted))
        except Exception as e:
            print(f"Error combining similarities: {e}")
            return max(0.0, min(1.0, base_sim))
    
    # Helper methods with error handling
    def _is_meaningful_token(self, token) -> bool:
        """Check if token is meaningful for analysis."""
        try:
            return (not token.is_stop and not token.is_punct and 
                    not token.is_space and len(token.text) > 1)
        except:
            return len(token.text) > 1 and token.text.isalpha()
    
    def _get_enhanced_token_sentiment(self, token, context: str) -> Dict:
        """Enhanced token sentiment with better caching."""
        try:
            cache_key = f"{token.text}_{token.pos_}_{hash(context[:50])}"
            
            if cache_key in self.cache['sentiment']:
                return self.cache['sentiment'][cache_key]
            
            pos_mapping = {
                'NOUN': wn.NOUN, 'PROPN': wn.NOUN,
                'VERB': wn.VERB, 'AUX': wn.VERB,
                'ADJ': wn.ADJ, 'ADV': wn.ADV
            }
            
            wn_pos = pos_mapping.get(token.pos_)
            if not wn_pos:
                return {}
            
            synset = lesk(context, token.text, wn_pos)
            
            if synset:
                swn_synset = swn.senti_synset(synset.name())
                
                sentiment_data = {
                    'pos_score': swn_synset.pos_score(),
                    'neg_score': swn_synset.neg_score(),
                    'obj_score': swn_synset.obj_score(),
                    'definition': synset.definition()
                }
                
                self.cache['sentiment'][cache_key] = sentiment_data
                return sentiment_data
            
        except Exception as e:
            print(f"Error getting token sentiment for {token.text}: {e}")
        
        return {}
    
    def _get_negation_context(self, token) -> Dict:
        """Enhanced negation context detection."""
        try:
            cache_key = f"neg_{token.i}_{token.sent.start}"
            
            if cache_key in self.cache['negation']:
                return self.cache['negation'][cache_key]
            
            is_negated = False
            negation_words = []
            
            # Check for negation in various contexts
            for child in token.children:
                if child.dep_ == 'neg' or child.text.lower() in ['not', 'no', 'never']:
                    is_negated = True
                    negation_words.append(child.text)
            
            # Check ancestors
            for ancestor in token.ancestors:
                for child in ancestor.children:
                    if child.dep_ == 'neg' and child.i < token.i:
                        is_negated = True
                        negation_words.append(child.text)
            
            context = {
                'is_negated': is_negated,
                'negation_words': negation_words,
                'strength': len(negation_words)
            }
            
            self.cache['negation'][cache_key] = context
            return context
        except Exception as e:
            print(f"Error getting negation context: {e}")
            return {'is_negated': False, 'negation_words': [], 'strength': 0}
    
    def _find_negation_scope(self, negation_token):
        """Find tokens affected by negation."""
        try:
            affected = []
            
            # Look at siblings and children
            if negation_token.head:
                for child in negation_token.head.children:
                    if child.i > negation_token.i:
                        affected.append(child)
            
            return affected
        except Exception as e:
            print(f"Error finding negation scope: {e}")
            return []
    
    def _get_dominant_sentiment(self, scores: Dict) -> str:
        """Get dominant sentiment from VADER scores."""
        try:
            if scores['compound'] >= 0.05:
                return 'positive'
            elif scores['compound'] <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _calculate_enhanced_token_similarity(self, answer_doc, criteria_doc, 
                                           original_answer: str, sentiment_analysis: Dict) -> Tuple[float, List[Dict], Dict]:
        """Enhanced token similarity with sentiment context."""
        try:
            answer_tokens = [t for t in answer_doc if self._is_meaningful_token(t)]
            criteria_tokens = [t for t in criteria_doc if self._is_meaningful_token(t)]
            
            if not criteria_tokens:
                return 0.0, [], {}
            
            total_score = 0
            similar_words = []
            match_counts = {'exact': 0, 'lemma': 0, 'semantic': 0}
            
            # Apply sentiment penalty for negative answers
            sentiment_penalty = 0.8 if sentiment_analysis['is_negative'] else 1.0
            
            for c_token in criteria_tokens:
                best_score = 0
                best_match = None
                
                for a_token in answer_tokens:
                    exact_match = 1.0 if c_token.text == a_token.text else 0
                    lemma_match = 0.8 if c_token.lemma_ == a_token.lemma_ else 0
                    semantic_match = self._calculate_semantic_similarity(c_token, a_token)
                    
                    token_score = max(exact_match, lemma_match, semantic_match) * sentiment_penalty
                    
                    if token_score > best_score:
                        best_score = token_score
                        best_match = a_token
                        
                        if exact_match > 0:
                            match_counts['exact'] += 1
                        elif lemma_match > 0:
                            match_counts['lemma'] += 1
                        elif semantic_match > 0:
                            match_counts['semantic'] += 1
                
                total_score += best_score
                
                if best_match and best_score > 0.3:
                    word_info = self._get_word_info(best_match, original_answer)
                    if word_info:
                        similar_words.append(word_info)
            
            avg_similarity = total_score / len(criteria_tokens) if criteria_tokens else 0
            
            match_details = {
                'exact_matches': match_counts['exact'],
                'lemma_matches': match_counts['lemma'], 
                'semantic_matches': match_counts['semantic'],
                'total_criteria_keywords': len(criteria_tokens),
                'sentiment_penalty_applied': sentiment_penalty < 1.0
            }
            
            return avg_similarity, similar_words, match_details
        except Exception as e:
            print(f"Error in token similarity calculation: {e}")
            return 0.0, [], {}
    
    def _calculate_semantic_similarity(self, token1, token2) -> float:
        """Calculate semantic similarity using WordNet."""
        try:
            synsets1 = wn.synsets(token1.text, pos=self._get_wordnet_pos(token1.pos_))
            synsets2 = wn.synsets(token2.text, pos=self._get_wordnet_pos(token2.pos_))
            
            if not synsets1 or not synsets2:
                return 0.0
            
            max_sim = 0
            for s1 in synsets1[:3]:
                for s2 in synsets2[:3]:
                    try:
                        sim = s1.path_similarity(s2)
                        if sim and sim > max_sim:
                            max_sim = sim
                    except:
                        continue
            
            return max_sim * 0.6
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _get_wordnet_pos(self, spacy_pos: str):
        """Map spaCy POS to WordNet POS."""
        try:
            pos_map = {
                'NOUN': wn.NOUN, 'PROPN': wn.NOUN,
                'VERB': wn.VERB, 'AUX': wn.VERB,
                'ADJ': wn.ADJ, 'ADV': wn.ADV
            }
            return pos_map.get(spacy_pos)
        except:
            return None
    
    def _get_word_info(self, token, context: str) -> Dict:
        """Get comprehensive word information."""
        try:
            wn_pos = self._get_wordnet_pos(token.pos_)
            if wn_pos:
                synset = lesk(context, token.text, wn_pos)
                if synset:
                    return {
                        'word': token.text,
                        'gloss': synset.definition(),
                        'negated': self._get_negation_context(token)['is_negated']
                    }
        except Exception as e:
            print(f"Error getting word info: {e}")
        
        return {
            'word': token.text,
            'gloss': 'No definition available',
            'negated': False
        }


# Updated SimilarityAssessmentApp class
class EnhancedSimilarityAssessmentApp:
    def __init__(self):
        self.app = flask.Flask(__name__)
        self.analyzer = OptimizedDynamicTextAnalyzer()
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
                return f"<h1>Enhanced Assessment Tool</h1><p>Question: {self.question}</p><p>Error loading template: {str(e)}</p>"
        
        @self.app.route('/health')
        def health_check():
            return {'status': 'healthy', 'message': 'Enhanced application is running'}
        
        @self.app.route('/assess', methods=['POST'])
        def assess():
            try:
                return self._process_enhanced_assessment()
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return f"<h1>Error Processing Assessment</h1><p>Error: {str(e)}</p><pre>{error_details}</pre><a href='/'>Back to Home</a>"
    
    def _process_enhanced_assessment(self):
        """Enhanced assessment processing with better negative detection."""
        try:
            # Get form data
            answer = flask.request.form['answer']
            score_given = int(flask.request.form['score'])
            
            # Enhanced sentiment analysis
            sentiment_analysis = self.analyzer.analyze_comprehensive_sentiment(answer)
            
            # Transform for template compatibility
            transformed_sentiment = self._transform_enhanced_sentiment(sentiment_analysis)
            
            # Enhanced similarity calculation
            similarity_results = self._compute_enhanced_similarity_results(answer)
            
            # Find best match with sentiment consideration
            best_match_data = self._find_enhanced_best_match(similarity_results, sentiment_analysis)
            
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
            print(f"Error in _process_enhanced_assessment: {e}")
            return f"<h1>Error Processing Assessment</h1><p>Error: {str(e)}</p><pre>{error_details}</pre><a href='/'>Back to Home</a>"
    
    def _compute_enhanced_similarity_results(self, answer):
        """Enhanced similarity computation with sentiment awareness."""
        results = {}
        
        for score, description in self.criteria.items():
            similarity, similar_words, criteria_word_count, match_details = (
                self.analyzer.calculate_enhanced_similarity(answer, description)
            )
            
            results[score] = {
                'similarity': similarity,
                'similar_words': similar_words,
                'criteria_word_count': criteria_word_count,
                'match_details': match_details
            }
        
        # Enhanced match percentage calculation
        similarities = [results[score]['similarity'] for score in results]
        max_similarity = max(similarities) if similarities else 0
        
        for score in results:
            if max_similarity > 0:
                results[score]['match_percentage'] = (results[score]['similarity'] / max_similarity) * 100
            else:
                results[score]['match_percentage'] = 0
        
        return results
    
    def _find_enhanced_best_match(self, similarity_results, sentiment_analysis):
        """Enhanced best match finding with sentiment consideration."""
        
        # If answer is strongly negative, bias towards lower scores
        if sentiment_analysis['is_negative'] and sentiment_analysis['negative_strength'] > 0.3:
            # Filter to lower scores (1-3) for negative answers
            filtered_results = {k: v for k, v in similarity_results.items() if k <= 3}
            if filtered_results:
                similarity_results = filtered_results
        
        sorted_scores = sorted(similarity_results.items(), 
                             key=lambda x: x[1]['similarity'], reverse=True)
        
        best_score = sorted_scores[0][0]
        
        return {
            'best_score': best_score,
            'best_similarity': f"{similarity_results[best_score]['similarity']:.4f}",
            'best_match_percentage': f"{similarity_results[best_score]['match_percentage']:.2f}",
            'similar_words': similarity_results[best_score].get('similar_words', []),
            'match_details': similarity_results[best_score]['match_details'],
            'sentiment_influenced': sentiment_analysis['is_negative']
        }
    
    def _transform_enhanced_sentiment(self, sentiment_analysis: Dict) -> Dict:
        """Transform enhanced sentiment analysis for template."""
        vader_data = sentiment_analysis.get('vader_analysis', {}).get('overall_scores', {})
        
        return {
            'overall_sentiment': self._get_sentiment_label(sentiment_analysis),
            'sentiment_scores': {
                'positive': int(vader_data.get('pos', 0) * 10),
                'negative': int(vader_data.get('neg', 0) * 10),
                'neutral': int(vader_data.get('neu', 0) * 10)
            },
            'context_sentiment': {
                'positive_contexts': self._extract_positive_contexts(sentiment_analysis),
                'negative_contexts': self._extract_negative_contexts(sentiment_analysis),
                'neutral_contexts': []
            },
            'word_sentiments': sentiment_analysis.get('sentiwordnet_analysis', {}).get('word_analyses', []),
            'confidence': sentiment_analysis.get('confidence', 0),
            'is_negative': sentiment_analysis.get('is_negative', False),
            'overall_polarity': sentiment_analysis.get('overall_polarity', 0),
            'negative_strength': sentiment_analysis.get('negative_strength', 0)
        }
    
    def _get_sentiment_label(self, sentiment_analysis: Dict) -> str:
        """Get enhanced sentiment label."""
        if sentiment_analysis.get('is_negative', False):
            return 'Negative'
        elif sentiment_analysis.get('overall_polarity', 0) > 0.1:
            return 'Positive'
        else:
            return 'Neutral'
    
    def _extract_positive_contexts(self, sentiment_analysis: Dict) -> List[str]:
        """Extract positive contexts from enhanced sentiment analysis."""
        word_analyses = sentiment_analysis.get('sentiwordnet_analysis', {}).get('word_analyses', [])
        positive_words = [w['word'] for w in word_analyses if w.get('pos_score', 0) > 0.1]
        return [f"Positive indicators: {', '.join(positive_words)}"] if positive_words else []
    
    def _extract_negative_contexts(self, sentiment_analysis: Dict) -> List[str]:
        """Extract negative contexts from enhanced sentiment analysis."""
        contexts = []
        
        # From word analysis
        word_analyses = sentiment_analysis.get('sentiwordnet_analysis', {}).get('word_analyses', [])
        negative_words = [w['word'] for w in word_analyses if w.get('neg_score', 0) > 0.1 or w.get('negation_context', {}).get('is_negated', False)]
        
        if negative_words:
            contexts.append(f"Negative indicators: {', '.join(negative_words)}")
        
        # From negative indicators
        negative_indicators = sentiment_analysis.get('negative_indicators', {})
        for category, words in negative_indicators.get('categories', {}).items():
            if words:
                contexts.append(f"Negative {category}: {', '.join(words)}")
        
        # From negation patterns
        negation_analysis = sentiment_analysis.get('negation_analysis', {})
        negation_patterns = negation_analysis.get('patterns', [])
        if negation_patterns:
            negation_words = [p['negation_word'] for p in negation_patterns]
            contexts.append(f"Negation patterns: {', '.join(negation_words)}")
        
        return contexts
    
    def _get_answer_glosses(self, answer: str) -> List[Dict]:
        """Get word definitions for the answer."""
        doc = self.analyzer.nlp(answer)
        glosses = []
        
        for token in doc:
            if self.analyzer._is_meaningful_token(token):
                word_info = self.analyzer._get_word_info(token, answer)
                if word_info and word_info['gloss'] != 'No definition available':
                    glosses.append({
                        'word': word_info['word'],
                        'pos': token.pos_,
                        'gloss': word_info['gloss']
                    })
        
        return glosses
    
    def run(self, debug=True):
        """Run the enhanced Flask application."""
        self.app.run(debug=debug)


# Initialize enhanced app
enhanced_similarity_app = EnhancedSimilarityAssessmentApp()
app = enhanced_similarity_app.app

# Main Execution
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)