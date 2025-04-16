import spacy
import networkx as nx
import torch
from torch_geometric.nn import GATConv
from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Hugging Face for Named Entity Recognition (NER)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

# GAT Model Definition
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index)
        return x

# Feature Extraction
def extract_features(tweet):
    # Dependency Parsing
    doc = nlp(tweet)
    dependencies = [(token.text, token.head.text) for token in doc]

    # Extract Named Entities
    entities = ner_pipeline(tweet)

    return dependencies, entities

# Graph Construction
def build_graph(dependencies, entities):
    G = nx.DiGraph()
    for word, head in dependencies:
        G.add_edge(head, word)
    for entity in entities:
        G.add_node(entity["word"], label=entity["entity"])
    return G

# Generate Summary using Pegasus
def generate_summary(tweet):
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Encode the input
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to("cpu")
    attention_mask = inputs["attention_mask"].to("cpu")

    # Generate summary with Pegasus
    output = pegasus_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=30,  # Adjust max_length for shorter summaries
        num_beams=4,    # Use beam search for better quality
        early_stopping=True,
        temperature=1.0  # Adjust temperature for randomness
    )
    
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Optional: Post-process the summary
    
    
    return summary
# Main Pipeline
def main_pipeline(tweet):
    # Step 1: Extract Features
    dependencies, entities = extract_features(tweet)

    # Step 2: Build Graph
    graph = build_graph(dependencies, entities)

    # Create a mapping from node labels to indices
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}

    # Step 3: Process Graph with GAT
    node_features = torch.randn(len(graph.nodes), 16)  # Random node embeddings
    edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in graph.edges()], dtype=torch.long).t().contiguous()
    
    gat_model = GAT(input_dim=16, hidden_dim=32, output_dim=16)
    graph_embedding = gat_model(node_features, edge_index)

    # Step 4: Generate Cause with Pegasus
    # Instead of passing graph_embedding, we focus on the tweet for clarity
    cause = generate_summary(tweet)
    return cause

# Example Execution
if __name__ == '__main__':
    tweet = "My day is good as i met my girl and then a bit sad as i can't stay there for long time "
    print("Original Tweet:", tweet)
    cause = main_pipeline(tweet)
    print("Generated Cause:", cause)