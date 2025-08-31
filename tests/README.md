# Tests for Subway Emergency Ontology System

This directory contains test files for all major components of the system.

## Test Structure

- `test_ontology.py` - Tests for ontology management and reasoning
- `test_nlp.py` - Tests for NLP components (entity extraction, intent classification)
- `test_reasoning.py` - Tests for dependency analysis and cascade prediction
- `conftest.py` - Shared test fixtures and configuration

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_ontology.py -v
python -m pytest tests/test_nlp.py -v
python -m pytest tests/test_reasoning.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Data

Test files use sample data located in `tests/data/` to avoid dependencies on production ontology files.
