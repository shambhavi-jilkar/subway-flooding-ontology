import spacy
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized: Optional[str] = None

class SubwayNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Subway-specific entity mappings
        self.infrastructure_map = {
            'pump a7': 'Pump_A7',
            'pump_a7': 'Pump_A7', 
            'pump room': 'Pump_A7',
            'signal hub 4b': 'Signal_Hub_4B',
            'signal_hub_4b': 'Signal_Hub_4B',
            'signal box': 'Signal_Hub_4B',
            'tunnel section b': 'Tunnels',
            'section b': 'Tunnels',
            'drainage system': 'Drainage_Systems',
            'pump rooms': 'Pump_Rooms',
            'power supply': 'Power_Supply'
        }
        
        self.hazard_map = {
            'water rising': 'FLOOD',
            'flooding': 'FLOOD',
            'flood': 'FLOOD',
            'water': 'FLOOD',
            'electrical failure': 'ELECTRICAL',
            'power outage': 'ELECTRICAL',
            'sparking': 'ELECTRICAL',
            'fire': 'FIRE',
            'smoke': 'FIRE'
        }
        
        self.severity_indicators = {
            'critical': 0.9,
            'urgent': 0.8,
            'immediate': 0.85,
            'emergency': 0.9,
            'fast': 0.7,
            'rapid': 0.7,
            'severe': 0.8,
            'major': 0.75,
            'minor': 0.3,
            'routine': 0.2
        }
        
        # Compile regex patterns for efficiency
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for entity extraction"""
        patterns = {
            'infrastructure': [
                re.compile(r'\b(pump|signal|power)\s*([A-Z]\d+)\b', re.IGNORECASE),
                re.compile(r'\b(tunnel|section)\s+([A-Z]\d?)\b', re.IGNORECASE),
                re.compile(r'\bpump\s+room\b', re.IGNORECASE),
                re.compile(r'\bsignal\s+(?:hub|box)\b', re.IGNORECASE)
            ],
            'hazard': [
                re.compile(r'\bwater\s+rising\b', re.IGNORECASE),
                re.compile(r'\b(?:electrical\s+)?(?:failure|outage)\b', re.IGNORECASE),
                re.compile(r'\b(?:flood|flooding)\b', re.IGNORECASE),
                re.compile(r'\bspark(?:ing|s)?\b', re.IGNORECASE)
            ],
            'severity': [
                re.compile(r'\b(critical|urgent|immediate|emergency)\b', re.IGNORECASE),
                re.compile(r'\b(fast|rapid|quick)\b', re.IGNORECASE),
                re.compile(r'\bASAP\b', re.IGNORECASE)
            ],
            'location': [
                re.compile(r'\b(?:tunnel\s+)?section\s+([A-Z]\d?)\b', re.IGNORECASE),
                re.compile(r'\btunnel\s+([A-Z]\d?)\b', re.IGNORECASE),
                re.compile(r'\bstation\s+(\w+)\b', re.IGNORECASE)
            ]
        }
        return patterns
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract all entities from text"""
        entities = []
        
        # Standard spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "PRODUCT", "FAC"]:
                mapped = self._map_to_ontology(ent.text)
                if mapped:
                    entities.append(Entity(
                        text=ent.text,
                        label="INFRASTRUCTURE",
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,
                        normalized=mapped
                    ))
        
        # Custom pattern matching
        entities.extend(self._extract_by_patterns(text))
        
        # Resolve conflicts and normalize
        entities = self._resolve_conflicts(entities)
        
        return entities
    
    def _extract_by_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    confidence = 0.9  # High confidence for pattern matches
                    normalized = None
                    
                    if entity_type == 'infrastructure':
                        normalized = self._normalize_infrastructure(match.group())
                    elif entity_type == 'hazard':
                        normalized = self._normalize_hazard(match.group())
                    elif entity_type == 'severity':
                        normalized = self._normalize_severity(match.group())
                    
                    entities.append(Entity(
                        text=match.group(),
                        label=entity_type.upper(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        normalized=normalized
                    ))
        
        return entities
    
    def _normalize_infrastructure(self, text: str) -> Optional[str]:
        """Normalize infrastructure mentions to ontology entities"""
        text_lower = text.lower().strip()
        
        # Direct mapping
        if text_lower in self.infrastructure_map:
            return self.infrastructure_map[text_lower]
        
        # Fuzzy matching for common variations
        if 'pump' in text_lower:
            if 'a7' in text_lower or '7' in text_lower:
                return 'Pump_A7'
            elif 'room' in text_lower:
                return 'Pump_A7'  # Assume pump room refers to main pump
        
        if 'signal' in text_lower:
            if '4b' in text_lower or 'hub' in text_lower:
                return 'Signal_Hub_4B'
        
        if 'tunnel' in text_lower or 'section' in text_lower:
            return 'Tunnels'
        
        return None
    
    def _normalize_hazard(self, text: str) -> Optional[str]:
        """Normalize hazard mentions"""
        text_lower = text.lower().strip()
        
        for key, value in self.hazard_map.items():
            if key in text_lower:
                return value
        
        return None
    
    def _normalize_severity(self, text: str) -> Optional[float]:
        """Normalize severity indicators to numeric scores"""
        text_lower = text.lower().strip()
        
        for indicator, score in self.severity_indicators.items():
            if indicator in text_lower:
                return score
        
        return None
    
    def _map_to_ontology(self, text: str) -> Optional[str]:
        """Map extracted text to ontology entities"""
        text_lower = text.lower().strip()
        return self.infrastructure_map.get(text_lower)
    
    def _resolve_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities, preferring higher confidence"""
        entities.sort(key=lambda x: (x.start, -x.confidence))
        resolved = []
        
        for entity in entities:
            # Check for overlap with existing entities
            overlapping = False
            for existing in resolved:
                if (entity.start < existing.end and entity.end > existing.start):
                    if entity.confidence <= existing.confidence:
                        overlapping = True
                        break
                    else:
                        # Remove lower confidence entity
                        resolved.remove(existing)
                        break
            
            if not overlapping:
                resolved.append(entity)
        
        return resolved
    
    def get_extraction_summary(self, entities: List[Entity]) -> Dict:
        """Create summary of extracted entities"""
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'infrastructure_assets': [],
            'hazards': [],
            'severity_scores': [],
            'confidence_avg': 0.0
        }
        
        for entity in entities:
            # Count by type
            if entity.label not in summary['by_type']:
                summary['by_type'][entity.label] = 0
            summary['by_type'][entity.label] += 1
            
            # Collect specific types
            if entity.label == 'INFRASTRUCTURE' and entity.normalized:
                summary['infrastructure_assets'].append(entity.normalized)
            elif entity.label == 'HAZARD' and entity.normalized:
                summary['hazards'].append(entity.normalized)
            elif entity.label == 'SEVERITY' and entity.normalized:
                summary['severity_scores'].append(entity.normalized)
        
        # Calculate average confidence
        if entities:
            summary['confidence_avg'] = sum(e.confidence for e in entities) / len(entities)
        
        return summary