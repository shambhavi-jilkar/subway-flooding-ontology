from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class IntentResult:
    intent: str
    confidence: float
    urgency_score: float
    features: Dict[str, float]

class IntentClassifier:
    def __init__(self):
        self.intent_keywords = {
            'EMERGENCY_REPORT': {
                'primary': ['emergency', 'critical', 'urgent', 'failure', 'flooding', 'help'],
                'secondary': ['fast', 'immediate', 'ASAP', 'now', 'problem', 'issue'],
                'urgency': ['critical', 'urgent', 'emergency', 'immediate', 'ASAP']
            },
            'STATUS_UPDATE': {
                'primary': ['status', 'update', 'report', 'current', 'situation'],
                'secondary': ['currently', 'now', 'present', 'condition'],
                'urgency': ['routine', 'normal', 'standard']
            },
            'INFORMATION_REQUEST': {
                'primary': ['information', 'details', 'clarification', 'status'],
                'secondary': ['need', 'require', 'request', 'want', 'check'],
                'urgency': ['when', 'where', 'what', 'how', 'why']
            }
        }
        
        # Initialize ML pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        self.is_trained = False
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate synthetic training data for intent classification"""
        training_texts = []
        training_labels = []
        
        # Emergency report examples
        emergency_examples = [
            "Water rising fast in tunnel section B near pump room",
            "Emergency! Pump A7 flooding critical situation",
            "Need immediate help electrical failure in signal hub",
            "URGENT: Major flooding in drainage system",
            "Critical situation pump room sparking need backup ASAP",
            "Emergency response required signal failure",
            "Help needed immediately water levels rising rapidly",
            "Electrical panel sparking critical safety issue"
        ]
        
        # Status update examples  
        status_examples = [
            "Current status of pump A7 operational",
            "Routine update tunnel section B normal conditions",
            "Status report all systems functioning normally",
            "Update on maintenance work signal hub operating",
            "Current situation stable no issues reported",
            "Regular status check drainage systems clear",
            "Operational update all equipment working properly",
            "Maintenance status report completed successfully"
        ]
        
        # Information request examples
        info_examples = [
            "What is the status of signal hub 4B?",
            "Need information on pump A7 capacity",
            "Can you clarify the maintenance schedule?",
            "Where is the nearest emergency equipment?",
            "How long will the repair take?",
            "What's the backup procedure for power failure?",
            "Need details on evacuation routes",
            "When is the next inspection scheduled?"
        ]
        
        # Combine training data
        training_texts.extend(emergency_examples)
        training_labels.extend(['EMERGENCY_REPORT'] * len(emergency_examples))
        
        training_texts.extend(status_examples)
        training_labels.extend(['STATUS_UPDATE'] * len(status_examples))
        
        training_texts.extend(info_examples)
        training_labels.extend(['INFORMATION_REQUEST'] * len(info_examples))
        
        # Train the pipeline
        self.pipeline.fit(training_texts, training_labels)
        self.is_trained = True
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and semantic features from text"""
        features = {}
        
        # Basic linguistic features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['urgent_caps'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Punctuation patterns
        features['multiple_exclamations'] = len(re.findall(r'!{2,}', text))
        features['ellipsis_count'] = text.count('...')
        
        # Keyword-based features
        text_lower = text.lower()
        for intent, keywords in self.intent_keywords.items():
            for category, words in keywords.items():
                feature_name = f'{intent}_{category}_count'
                features[feature_name] = sum(1 for word in words if word in text_lower)
        
        # Urgency indicators
        urgency_words = ['critical', 'urgent', 'immediate', 'emergency', 'fast', 'ASAP', 'help', 'now']
        features['urgency_score'] = sum(1 for word in urgency_words if word.lower() in text_lower)
        
        # Time indicators
        time_words = ['now', 'immediate', 'soon', 'later', 'when', 'scheduled']
        features['time_reference'] = sum(1 for word in time_words if word.lower() in text_lower)
        
        return features
    
    def classify_intent(self, text: str) -> IntentResult:
        """Classify intent of input text"""
        if not self.is_trained:
            raise ValueError("Intent classifier not trained")
        
        # Extract features
        features = self.extract_features(text)
        
        # Get ML prediction
        intent_proba = self.pipeline.predict_proba([text])[0]
        intent_classes = self.pipeline.classes_
        
        # Find best prediction
        best_idx = np.argmax(intent_proba)
        ml_intent = intent_classes[best_idx]
        ml_confidence = intent_proba[best_idx]
        
        # Rule-based refinement
        rule_intent, rule_confidence = self._apply_rules(text, features)
        
        # Combine ML and rule-based results
        if rule_confidence > 0.8 and rule_confidence > ml_confidence:
            final_intent = rule_intent
            final_confidence = rule_confidence
        else:
            final_intent = ml_intent
            final_confidence = ml_confidence
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency(text, features)
        
        return IntentResult(
            intent=final_intent,
            confidence=final_confidence,
            urgency_score=urgency_score,
            features=features
        )
    
    def _apply_rules(self, text: str, features: Dict[str, float]) -> Tuple[str, float]:
        """Apply rule-based classification as backup/refinement"""
        text_lower = text.lower()
        
        # Emergency indicators
        emergency_score = 0
        if features['urgency_score'] >= 2:
            emergency_score += 0.3
        if features['exclamation_count'] >= 1:
            emergency_score += 0.2
        if features['urgent_caps'] >= 1:
            emergency_score += 0.2
        if any(word in text_lower for word in ['help', 'emergency', 'critical', 'failure']):
            emergency_score += 0.3
        
        # Question indicators
        question_score = 0
        if features['question_count'] >= 1:
            question_score += 0.4
        if any(word in text_lower for word in ['what', 'where', 'when', 'how', 'why']):
            question_score += 0.3
        if 'information' in text_lower or 'details' in text_lower:
            question_score += 0.3
        
        # Status update indicators
        status_score = 0
        if any(word in text_lower for word in ['status', 'update', 'report', 'current']):
            status_score += 0.4
        if features['urgency_score'] == 0 and features['exclamation_count'] == 0:
            status_score += 0.2
        
        # Determine best rule-based classification
        scores = {
            'EMERGENCY_REPORT': emergency_score,
            'INFORMATION_REQUEST': question_score,
            'STATUS_UPDATE': status_score
        }
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        return best_intent, min(best_score, 1.0)
    
    def _calculate_urgency(self, text: str, features: Dict[str, float]) -> float:
        """Calculate urgency score from 0.0 to 1.0"""
        urgency = 0.0
        text_lower = text.lower()
        
        # Keyword-based urgency
        high_urgency = ['critical', 'emergency', 'urgent', 'immediate']
        medium_urgency = ['fast', 'quick', 'soon', 'ASAP']
        low_urgency = ['routine', 'scheduled', 'normal', 'standard']
        
        for word in high_urgency:
            if word.lower() in text_lower:
                urgency += 0.3
        
        for word in medium_urgency:
            if word.lower() in text_lower:
                urgency += 0.2
        
        for word in low_urgency:
            if word.lower() in text_lower:
                urgency -= 0.2
        
        # Punctuation-based urgency
        urgency += features['exclamation_count'] * 0.1
        urgency += features['urgent_caps'] * 0.15
        urgency += features['multiple_exclamations'] * 0.2
        
        # Context-based urgency
        hazard_words = ['flooding', 'fire', 'electrical', 'failure', 'sparking']
        for word in hazard_words:
            if word in text_lower:
                urgency += 0.2
        
        help_words = ['help', 'assistance', 'backup', 'support']
        for word in help_words:
            if word in text_lower:
                urgency += 0.15
        
        return min(urgency, 1.0)

# SPARQL Query Generator
class SPARQLQueryGenerator:
    def __init__(self):
        self.base_prefixes = """
        PREFIX subway: <http://subway-emergency.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        """
        
        self.query_templates = {
            'IMPACT_ASSESSMENT': """
                {prefixes}
                SELECT ?asset ?dependency ?impact_level ?passenger_count ?label WHERE {{
                    subway:{asset_id} ?property ?dependency .
                    ?dependency rdfs:label ?label .
                    OPTIONAL {{ ?dependency subway:criticalityScore ?impact_level }}
                    OPTIONAL {{ ?dependency subway:dailyPassengerCount ?passenger_count }}
                    FILTER(?property = subway:powers || ?property = subway:controls || ?property = subway:affects)
                }}
                ORDER BY DESC(?impact_level) DESC(?passenger_count)
            """,
            
            'DEPENDENCY_ANALYSIS': """
                {prefixes}
                SELECT ?dependent ?relationship ?depth WHERE {{
                    subway:{asset_id} subway:affects* ?dependent .
                    subway:{asset_id} ?relationship ?dependent .
                    ?dependent rdfs:label ?label .
                    OPTIONAL {{ ?dependent subway:criticalityScore ?score }}
                }}
                ORDER BY DESC(?score)
            """,
            
            'STATUS_CHECK': """
                {prefixes}
                SELECT ?asset ?status ?score ?connections WHERE {{
                    subway:{asset_id} rdfs:label ?asset .
                    OPTIONAL {{ subway:{asset_id} subway:operationalStatus ?status }}
                    OPTIONAL {{ subway:{asset_id} subway:criticalityScore ?score }}
                    OPTIONAL {{ subway:{asset_id} subway:connectionCount ?connections }}
                }}
            """,
            
            'PROTOCOL_LOOKUP': """
                {prefixes}
                SELECT ?protocol ?action ?location WHERE {{
                    ?protocol subway:applicableToHazard subway:{hazard_type} .
                    OPTIONAL {{ ?protocol subway:recommendedAction ?action }}
                    OPTIONAL {{ ?protocol subway:equipmentLocation ?location }}
                }}
            """,
            
            'EVACUATION_ROUTES': """
                {prefixes}
                SELECT ?route ?safety_score ?time WHERE {{
                    ?route subway:fromLocation subway:{start_location} .
                    ?route subway:toLocation ?destination .
                    OPTIONAL {{ ?route subway:safetyScore ?safety_score }}
                    OPTIONAL {{ ?route subway:estimatedTime ?time }}
                    FILTER NOT EXISTS {{
                        ?route subway:hasActiveHazard ?hazard .
                        ?hazard subway:severityLevel ?severity .
                        FILTER(?severity >= 0.5)
                    }}
                }}
                ORDER BY DESC(?safety_score) ASC(?time)
            """
        }
    
    def generate_query(self, query_type: str, entities: Dict, intent_result: IntentResult) -> str:
        """Generate SPARQL query based on extracted entities and intent"""
        template = self.query_templates.get(query_type)
        if not template:
            return self._generate_generic_query(entities)
        
        # Prepare substitutions
        substitutions = {'prefixes': self.base_prefixes}
        
        # Map entities to query parameters
        if 'infrastructure_assets' in entities and entities['infrastructure_assets']:
            asset = entities['infrastructure_assets'][0].replace(' ', '_')
            substitutions['asset_id'] = asset
        
        if 'hazards' in entities and entities['hazards']:
            hazard = entities['hazards'][0]
            substitutions['hazard_type'] = hazard
        
        # Add default values if missing
        substitutions.setdefault('asset_id', 'Pump_A7')
        substitutions.setdefault('hazard_type', 'FLOOD')
        substitutions.setdefault('start_location', 'Tunnels')
        
        # Modify query based on urgency
        query = template.format(**substitutions)
        if intent_result.urgency_score >= 0.8:
            query = self._add_urgency_constraints(query)
        
        return query
    
    def _generate_generic_query(self, entities: Dict) -> str:
        """Generate a generic query when no specific template matches"""
        if entities.get('infrastructure_assets'):
            asset = entities['infrastructure_assets'][0].replace(' ', '_')
            return f"""
            {self.base_prefixes}
            SELECT ?property ?value ?label WHERE {{
                subway:{asset} ?property ?value .
                OPTIONAL {{ ?value rdfs:label ?label }}
                FILTER(?property != rdf:type)
            }}
            """
        else:
            return f"""
            {self.base_prefixes}
            SELECT ?asset ?label ?score WHERE {{
                ?asset subway:criticalityScore ?score .
                ?asset rdfs:label ?label .
            }}
            ORDER BY DESC(?score)
            LIMIT 10
            """
    
    def _add_urgency_constraints(self, query: str) -> str:
        """Add urgency-specific constraints to query"""
        # Add filters for high-priority items
        if 'ORDER BY' in query:
            # Add criticality filter before ORDER BY
            order_index = query.find('ORDER BY')
            before_order = query[:order_index]
            order_clause = query[order_index:]
            
            urgency_filter = """
            FILTER NOT EXISTS { ?dependency subway:operationalStatus "FAILED" }
            OPTIONAL { ?dependency subway:criticalityScore ?crit_score }
            """
            
            return before_order + urgency_filter + order_clause
        else:
            return query + """
            FILTER NOT EXISTS { ?asset subway:operationalStatus "FAILED" }
            """

# Complete NLP Pipeline
class NLPProcessor:
    def __init__(self):
        self.ner = SubwayNER()
        self.intent_classifier = IntentClassifier()
        self.query_generator = SPARQLQueryGenerator()
    
    def process_text(self, text: str) -> Dict:
        """Complete NLP processing pipeline"""
        # Step 1: Extract entities
        entities = self.ner.extract_entities(text)
        entity_summary = self.ner.get_extraction_summary(entities)
        
        # Step 2: Classify intent
        intent_result = self.intent_classifier.classify_intent(text)
        
        # Step 3: Generate appropriate SPARQL query
        query_type = self._determine_query_type(intent_result, entity_summary)
        sparql_query = self.query_generator.generate_query(
            query_type, entity_summary, intent_result
        )
        
        return {
            'original_text': text,
            'entities': [
                {
                    'text': e.text,
                    'label': e.label,
                    'confidence': e.confidence,
                    'normalized': e.normalized
                } for e in entities
            ],
            'entity_summary': entity_summary,
            'intent': {
                'classification': intent_result.intent,
                'confidence': intent_result.confidence,
                'urgency_score': intent_result.urgency_score
            },
            'sparql_query': sparql_query,
            'query_type': query_type,
            'processing_metadata': {
                'entity_count': len(entities),
                'avg_entity_confidence': entity_summary['confidence_avg'],
                'intent_confidence': intent_result.confidence
            }
        }
    
    def _determine_query_type(self, intent_result: IntentResult, entity_summary: Dict) -> str:
        """Determine which type of SPARQL query to generate"""
        if intent_result.intent == 'EMERGENCY_REPORT':
            if entity_summary['infrastructure_assets']:
                return 'IMPACT_ASSESSMENT'
            else:
                return 'DEPENDENCY_ANALYSIS'
        elif intent_result.intent == 'INFORMATION_REQUEST':
            if entity_summary['infrastructure_assets']:
                return 'STATUS_CHECK'
            else:
                return 'PROTOCOL_LOOKUP'
        elif intent_result.intent == 'STATUS_UPDATE':
            return 'STATUS_CHECK'
        else:
            return 'IMPACT_ASSESSMENT'  # Default fallback