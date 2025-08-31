import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class CascadeNode:
    asset_name: str
    failure_time: float
    impact_level: float
    affected_services: List[str]
    passenger_count: int

@dataclass
class CascadeAnalysis:
    primary_asset: str
    cascade_tree: Dict
    impact_timeline: List[CascadeNode]
    total_affected_services: Set[str]
    critical_path: List[str]
    total_passenger_impact: int

class DependencyAnalyzer:
    def __init__(self, ontology_manager):
        self.ontology = ontology_manager
        self.dependency_graph = nx.DiGraph()
        self.failure_time_models = {
            'POWER_DEPENDENCY': {'base_time': 15, 'variance': 5},
            'CONTROL_DEPENDENCY': {'base_time': 5, 'variance': 2},
            'SUPPORT_DEPENDENCY': {'base_time': 30, 'variance': 10},
            'INFORMATION_DEPENDENCY': {'base_time': 60, 'variance': 20}
        }
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """Build networkx graph from ontology relationships"""
        # Query all dependency relationships
        query = """
        PREFIX subway: <http://subway-emergency.org/ontology#>
        SELECT ?source ?target ?relationship ?source_label ?target_label WHERE {
            ?source ?relationship ?target .
            ?source rdfs:label ?source_label .
            ?target rdfs:label ?target_label .
            FILTER(?relationship = subway:powers || 
                   ?relationship = subway:controls || 
                   ?relationship = subway:affects ||
                   ?relationship = subway:dependsOn)
        }
        """
        
        results = self.ontology.graph.query(query)
        
        for row in results:
            source = str(row.source_label).replace(' ', '_')
            target = str(row.target_label).replace(' ', '_')
            relationship = str(row.relationship).split('#')[-1]
            
            # Add edge with relationship type
            self.dependency_graph.add_edge(
                source, target,
                relationship=relationship,
                weight=self._calculate_edge_weight(relationship)
            )
    
    def _calculate_edge_weight(self, relationship: str) -> float:
        """Calculate edge weight based on relationship type"""
        weights = {
            'powers': 0.9,      # High dependency
            'controls': 0.85,   # High dependency  
            'affects': 0.7,     # Medium dependency
            'dependsOn': 0.8    # High dependency
        }
        return weights.get(relationship, 0.5)
    
    def analyze_cascade_failure(self, failed_asset: str, time_horizon: int = 60) -> CascadeAnalysis:
        """Analyze potential cascade failures within time horizon"""
        cascade_nodes = []
        affected_services = set()
        
        # Find all reachable nodes from failed asset
        if failed_asset not in self.dependency_graph:
            return self._empty_cascade_analysis(failed_asset)
        
        # Use BFS to find cascade propagation
        visited = set()
        queue = [(failed_asset, 0, 1.0)]  # (node, time, impact_factor)
        
        while queue:
            current_node, current_time, impact_factor = queue.pop(0)
            
            if current_node in visited or current_time > time_horizon:
                continue
            
            visited.add(current_node)
            
            # Get node information from ontology
            node_info = self._get_node_info(current_node)
            
            # Calculate impact for this node
            if current_time > 0:  # Don't include the original failed asset
                cascade_node = CascadeNode(
                    asset_name=current_node,
                    failure_time=current_time,
                    impact_level=node_info['criticality'] * impact_factor,
                    affected_services=node_info['services'],
                    passenger_count=node_info['passengers']
                )
                cascade_nodes.append(cascade_node)
                affected_services.update(node_info['services'])
            
            # Find dependent nodes
            for neighbor in self.dependency_graph.successors(current_node):
                if neighbor not in visited:
                    edge_data = self.dependency_graph[current_node][neighbor]
                    
                    # Calculate failure time for dependent node
                    failure_time = self._estimate_failure_time(
                        current_node, neighbor, edge_data
                    )
                    
                    new_time = current_time + failure_time
                    new_impact = impact_factor * edge_data['weight']
                    
                    if new_time <= time_horizon:
                        queue.append((neighbor, new_time, new_impact))
        
        # Sort by failure time
        cascade_nodes.sort(key=lambda x: x.failure_time)
        
        # Find critical path (highest impact sequence)
        critical_path = self._find_critical_path(failed_asset, cascade_nodes)
        
        # Calculate total passenger impact
        total_passengers = sum(node.passenger_count for node in cascade_nodes)
        
        return CascadeAnalysis(
            primary_asset=failed_asset,
            cascade_tree=self._build_cascade_tree(cascade_nodes),
            impact_timeline=cascade_nodes,
            total_affected_services=affected_services,
            critical_path=critical_path,
            total_passenger_impact=total_passengers
        )
    
    def _get_node_info(self, node_name: str) -> Dict:
        """Get detailed information about a node from ontology"""
        node_uri = f"subway:{node_name}"
        
        query = f"""
        PREFIX subway: <http://subway-emergency.org/ontology#>
        SELECT ?criticality ?passengers ?status WHERE {{
            {node_uri} subway:criticalityScore ?criticality .
            OPTIONAL {{ {node_uri} subway:dailyPassengerCount ?passengers }}
            OPTIONAL {{ {node_uri} subway:operationalStatus ?status }}
        }}
        """
        
        results = list(self.ontology.graph.query(query))
        
        if results:
            row = results[0]
            return {
                'criticality': float(row.criticality) if row.criticality else 0.5,
                'passengers': int(row.passengers) if row.passengers else 0,
                'status': str(row.status) if row.status else 'UNKNOWN',
                'services': self._get_affected_services(node_name)
            }
        else:
            return {
                'criticality': 0.5,
                'passengers': 0,
                'status': 'UNKNOWN',
                'services': []
            }
    
    def _get_affected_services(self, node_name: str) -> List[str]:
        """Get services affected by this node"""
        # Simple heuristic based on node type and connections
        services = []
        
        if 'Signal' in node_name:
            services.extend(['Train_Control', 'Safety_Systems'])
        if 'Pump' in node_name:
            services.extend(['Drainage', 'Flood_Control'])
        if 'Power' in node_name:
            services.extend(['Electrical_Systems', 'Lighting', 'Communications'])
        if 'Tunnel' in node_name:
            services.extend(['Passenger_Transport', 'Emergency_Access'])
        
        return services
    
    def _estimate_failure_time(self, source: str, target: str, edge_data: Dict) -> float:
        """Estimate time until target fails given source failure"""
        relationship = edge_data['relationship']
        weight = edge_data['weight']
        
        # Get base failure time for relationship type
        failure_model = self.failure_time_models.get(
            f"{relationship.upper()}_DEPENDENCY",
            {'base_time': 20, 'variance': 5}
        )
        
        base_time = failure_model['base_time']
        variance = failure_model['variance']
        
        # Adjust based on relationship strength
        adjusted_time = base_time / weight
        
        # Add some randomness (in real system, this would be based on actual conditions)
        import random
        time_variation = random.uniform(-variance, variance)
        
        return max(1, adjusted_time + time_variation)
    
    def _find_critical_path(self, start_node: str, cascade_nodes: List[CascadeNode]) -> List[str]:
        """Find the path with highest cumulative impact"""
        if not cascade_nodes:
            return [start_node]
        
        # Simple heuristic: path through highest impact nodes
        sorted_nodes = sorted(cascade_nodes, key=lambda x: x.impact_level, reverse=True)
        
        critical_path = [start_node]
        for node in sorted_nodes[:3]:  # Take top 3 highest impact
            critical_path.append(node.asset_name)
        
        return critical_path
    
    def _build_cascade_tree(self, cascade_nodes: List[CascadeNode]) -> Dict:
        """Build hierarchical representation of cascade"""
        tree = {}
        
        for node in cascade_nodes:
            tree[node.asset_name] = {
                'failure_time': node.failure_time,
                'impact_level': node.impact_level,
                'services': node.affected_services,
                'passengers': node.passenger_count
            }
        
        return tree
    
    def _empty_cascade_analysis(self, asset: str) -> CascadeAnalysis:
        """Return empty analysis for unknown assets"""
        return CascadeAnalysis(
            primary_asset=asset,
            cascade_tree={},
            impact_timeline=[],
            total_affected_services=set(),
            critical_path=[asset],
            total_passenger_impact=0
        )
    
    def calculate_network_metrics(self) -> Dict:
        """Calculate network-wide dependency metrics"""
        metrics = {}
        
        # Centrality measures
        metrics['degree_centrality'] = nx.degree_centrality(self.dependency_graph)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.dependency_graph)
        metrics['closeness_centrality'] = nx.closeness_centrality(self.dependency_graph)
        
        # Network structure
        metrics['total_nodes'] = self.dependency_graph.number_of_nodes()
        metrics['total_edges'] = self.dependency_graph.number_of_edges()
        metrics['density'] = nx.density(self.dependency_graph)
        
        # Critical nodes (top 5 by different measures)
        metrics['most_connected'] = sorted(
            metrics['degree_centrality'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        metrics['most_central'] = sorted(
            metrics['betweenness_centrality'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return metrics

### **4.2 Real-Time Inference Engine**

**src/reasoning/inference_engine.py:**
```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class RuleType(Enum):
    DEPENDENCY = "dependency"
    ESCALATION = "escalation"
    CLASSIFICATION = "classification"
    PREDICTION = "prediction"

@dataclass
class InferenceRule:
    name: str
    rule_type: RuleType
    conditions: List[Dict]
    actions: List[Dict]
    confidence: float
    priority: int

@dataclass
class InferenceResult:
    rule_name: str
    triggered: bool
    confidence: float
    new_facts: List[Dict]
    recommendations: List[str]

class RealTimeInferenceEngine:
    def __init__(self, ontology_manager, dependency_analyzer):
        self.ontology = ontology_manager
        self.dependency_analyzer = dependency_analyzer
        self.rules = []
        self.working_memory = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize inference rules
        self._initialize_rules()# Dynamic Ontology Implementation - From Scratch
## Building Your NLP-Enhanced Emergency Communication System


