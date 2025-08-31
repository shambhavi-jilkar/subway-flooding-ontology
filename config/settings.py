from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Project Info
    project_name: str = "Subway Emergency Ontology"
    version: str = "1.0.0"
    
    # NLP Configuration
    spacy_model: str = "en_core_web_sm"
    max_text_length: int = 1000
    confidence_threshold: float = 0.7
    
    # Ontology Configuration
    ontology_file: str = "data/ontology/subway_emergency.ttl"
    sparql_endpoint: Optional[str] = None
    reasoning_enabled: bool = True
    
    # Performance
    cache_ttl: int = 300  # 5 minutes
    max_concurrent_requests: int = 50
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()