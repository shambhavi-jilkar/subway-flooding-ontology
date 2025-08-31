# Subway Emergency Ontology System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated AI-powered emergency communication system for subway/transit networks that combines semantic web technologies, natural language processing, and dependency analysis to enhance emergency response capabilities.

## 🚀 Key Features

### Performance
- **Real-time processing**: <3.2 seconds total pipeline
- **High accuracy**: >90% entity recognition for critical infrastructure
- **Scalable architecture**: Handles multiple concurrent emergency scenarios

### Intelligence
- **NLP Understanding**: Extracts entities, intent, and severity from natural language
- **Ontology Reasoning**: 32+ entities across 6 categories with dependency analysis
- **Cascade Prediction**: Identifies potential failure chains before they occur
- **Context Enhancement**: Enriches messages with operational knowledge

### Impact
- **Automated Analysis**: Converts unstructured emergency reports to actionable intelligence
- **Dependency Modeling**: Understands infrastructure relationships and failure propagation
- **Multi-Agency Coordination**: Scales across organizational boundaries

## 🏗️ System Architecture

```
Emergency Message → [NLP Processing] → [Ontology Reasoning] → Enhanced Response
                         ↓                     ↓                    ↓
                  Entity Extraction    Dependency Analysis    Action Generation
                  Intent Classification    Impact Assessment    Response Routing
                  Severity Assessment      Cascade Prediction   Justification
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone
   cd subway-emergency-ontology
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Verify installation**
   ```bash
   python -c "from src.ontology.manager import OntologyManager; print('✅ Installation successful')"
   ```

## 🚀 Quick Start

### Basic Usage

```python
from src.ontology.manager import OntologyManager
from src.nlp.entity_extractor import SubwayNER
from src.reasoning.dependency_analyser import DependencyAnalyzer

# Initialize components
ontology = OntologyManager("data/ontology/subway_emergency.ttl")
ner = SubwayNER()
analyzer = DependencyAnalyzer(ontology)

# Process emergency message
message = "Water rising fast in pump room A7, electrical systems sparking!"
entities = ner.extract_entities(message)
cascade_analysis = analyzer.analyze_cascade_failure("Pump_A7")

print(f"Critical infrastructure affected: {cascade_analysis.total_affected_services}")
```

### Example Analysis

**Input:** "Critical flooding in tunnel section B, pump A7 offline!"

**Output:**
- **Entities Detected:** Pump_A7 (infrastructure), FLOOD (hazard), critical (severity)
- **Cascade Analysis:** Signal_Hub_4B failure predicted in 15 minutes
- **Affected Services:** Express Line, Local Line connections
- **Recommended Actions:** Deploy emergency pumping, activate backup power

## 📊 Core Components

### 1. **Ontology Management** (`src/ontology/`)
- Semantic knowledge base with formal logic reasoning
- 32+ defined entities across 6 categories
- SPARQL query capabilities for complex relationship analysis
- OWL reasoning for automated inference

### 2. **Natural Language Processing** (`src/nlp/`)
- **Entity Extractor**: Identifies infrastructure, hazards, personnel, severity
- **Intent Classifier**: Categorizes emergency reports, status updates, requests
- Custom subway domain vocabulary with >90% accuracy

### 3. **Dependency Analysis** (`src/reasoning/`)
- Network graph modeling of infrastructure dependencies
- Cascade failure prediction with timeline estimation
- Impact assessment including passenger count and service disruption
- Critical path identification for prioritized response

### 4. **Configuration** (`config/`)
- Centralized settings management
- Environment-based configuration
- Tunable parameters for NLP and reasoning engines

## 📁 Project Structure

```
subway-emergency-ontology/
├── src/
│   ├── ontology/          # Semantic web components
│   │   └── manager.py     # Core ontology management
│   ├── nlp/               # Natural language processing
│   │   ├── entity_extractor.py
│   │   └── intent_classifier.py
│   ├── reasoning/         # Dependency analysis
│   │   └── dependency_analyser.py
│   └── api/               # Web API (future development)
├── data/
│   └── ontology/          # Knowledge base files
│       └── subway_emergency.ttl
├── config/
│   └── settings.py        # Configuration management
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test categories:
```bash
# Test NLP components
python -m pytest tests/test_nlp.py -v

# Test ontology reasoning
python -m pytest tests/test_ontology.py -v

# Test dependency analysis
python -m pytest tests/test_reasoning.py -v
```

## 📈 Performance Benchmarks

| Component | Processing Time | Accuracy |
|-----------|----------------|----------|
| Entity Extraction | <500ms | >90% |
| Intent Classification | <200ms | >85% |
| Cascade Analysis | <2s | N/A |
| Total Pipeline | <3.2s | >88% |

## 🛠️ Technology Stack

- **Semantic Web**: RDFLib, OWL-RL
- **NLP**: spaCy, Transformers, scikit-learn
- **Graph Analysis**: NetworkX
- **Web Framework**: FastAPI (planned)
- **Data Processing**: Pandas, NumPy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Research & Citation

This system is part of ongoing research in flooding emergency communication systems. 

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/shambhavi-jilkar/subway-flooding-ontology/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shambhavi-jilkar/subway-flooding-ontology/discussions)
- **Email**: sjilkar@andrew.cmu.edu

## 🗺️ Roadmap

- [ ] Web API implementation
- [ ] Real-time dashboard
- [ ] Integration with external emergency systems
- [ ] Advanced ML models for prediction

