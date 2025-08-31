from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import owlrl
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OntologyManager:
    def __init__(self, ontology_file: str):
        self.graph = Graph()
        self.subway_ns = Namespace("http://subway-emergency.org/ontology#")
        self.ontology_file = ontology_file
        
        # Bind namespaces
        self.graph.bind("subway", self.subway_ns)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        
        # Load ontology
        self.load_ontology()
        
        # Initialize reasoner
        self.reasoner = owlrl.CombinedClosure.RDFS_OWLRL_Closure(
            self.graph,
            improved_datatypes=True,
            rdfs_closure=True,
            owl_closure=True
        )
    
    def load_ontology(self):
        """Load ontology from file"""
        try:
            self.graph.parse(self.ontology_file, format='turtle')
            logger.info(f"Loaded ontology with {len(self.graph)} triples")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            raise
    
    def perform_reasoning(self) -> Dict[str, int]:
        """Execute inference rules and return statistics"""
        initial_size = len(self.graph)
        self.reasoner.closure()
        final_size = len(self.graph)
        
        return {
            'initial_triples': initial_size,
            'final_triples': final_size,
            'inferred_triples': final_size - initial_size
        }
    
    def get_entity_connections(self, entity_name: str) -> List[Dict]:
        """Get all connections for an entity"""
        entity_uri = self.subway_ns[entity_name]
        
        query = f"""
        SELECT ?property ?connected_entity ?direction WHERE {{
            {{
                <{entity_uri}> ?property ?connected_entity .
                BIND("outgoing" as ?direction)
            }} UNION {{
                ?connected_entity ?property <{entity_uri}> .
                BIND("incoming" as ?direction)
            }}
            FILTER(?property != rdf:type && ?property != rdfs:label)
        }}
        """
        
        results = []
        for row in self.graph.query(query):
            results.append({
                'property': str(row.property),
                'connected_entity': str(row.connected_entity),
                'direction': str(row.direction)
            })
        
        return results
    
    def get_critical_assets(self, threshold: float = 0.8) -> List[Dict]:
        """Get assets with criticality score above threshold"""
        query = f"""
        PREFIX subway: <http://subway-emergency.org/ontology#>
        SELECT ?asset ?label ?score ?connections WHERE {{
            ?asset subway:criticalityScore ?score .
            ?asset rdfs:label ?label .
            OPTIONAL {{ ?asset subway:connectionCount ?connections }}
            FILTER(?score >= {threshold})
        }}
        ORDER BY DESC(?score)
        """
        
        results = []
        for row in self.graph.query(query):
            results.append({
                'asset': str(row.asset),
                'label': str(row.label),
                'criticality_score': float(row.score),
                'connections': int(row.connections) if row.connections else 0
            })
        
        return results
    
    def find_dependencies(self, asset_name: str, max_depth: int = 3) -> Dict:
        """Find dependency chain for an asset"""
        asset_uri = self.subway_ns[asset_name]
        
        # Get direct dependencies
        query = f"""
        PREFIX subway: <http://subway-emergency.org/ontology#>
        SELECT ?dependent ?property ?label WHERE {{
            <{asset_uri}> ?property ?dependent .
            ?dependent rdfs:label ?label .
            FILTER(?property = subway:powers || ?property = subway:controls || ?property = subway:affects)
        }}
        """
        
        dependencies = {}
        for row in self.graph.query(query):
            dep_name = str(row.label)
            dependencies[dep_name] = {
                'uri': str(row.dependent),
                'relationship': str(row.property),
                'depth': 1
            }
            
            # Recursively find deeper dependencies
            if max_depth > 1:
                deeper_deps = self.find_dependencies(
                    dep_name.replace(" ", "_"), 
                    max_depth - 1
                )
                dependencies[dep_name]['dependencies'] = deeper_deps
        
        return dependencies
    
    def calculate_impact_score(self, asset_name: str) -> float:
        """Calculate comprehensive impact score for an asset"""
        dependencies = self.find_dependencies(asset_name)
        connections = self.get_entity_connections(asset_name)
        
        # Get passenger impact
        passenger_query = f"""
        PREFIX subway: <http://subway-emergency.org/ontology#>
        SELECT ?passengers WHERE {{
            subway:{asset_name} subway:affects* ?structure .
            ?structure subway:dailyPassengerCount ?passengers .
        }}
        """
        
        total_passengers = 0
        for row in self.graph.query(passenger_query):
            total_passengers += int(row.passengers)
        
        # Calculate composite score
        dependency_score = len(dependencies) * 0.3
        connection_score = len(connections) * 0.2  
        passenger_score = min(total_passengers / 100000, 1.0) * 0.5
        
        return min(dependency_score + connection_score + passenger_score, 1.0)