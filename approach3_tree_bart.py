import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertModel
from nltk import Tree, word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Initialize BERT and BART models and tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def extract_tree_structure(input_text):
    """
    Uses NLTK to create a tree-like structure of the input text based on syntax.
    """
    tokens = word_tokenize(input_text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return named_entities

def summarize_with_bart(input_text):
    """
    Generates a summary using BART.
    """
    inputs = bart_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=100, min_length=20, length_penalty=2.0, num_beams=4)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_tree_summary(input_text):
    """
    Generates a summary for each level of the tree and combines the results.
    """
    tree = extract_tree_structure(input_text)
    summaries = []

    def traverse_tree(t):
        if isinstance(t, Tree):
            # Combine leaf node text
            node_text = " ".join([word for word, _, _ in tree2conlltags(t)])
            summary = summarize_with_bart(node_text)
            summaries.append(summary)
            for subtree in t:
                traverse_tree(subtree)

    traverse_tree(tree)
    return summaries

def refine_with_rag(summaries):
    """
    Uses a mock RAG method to refine the summary. 
    (Actual RAG implementations require a retriever and a generator.)
    """
    refined_summary = " ".join(summaries)
    inputs = bart_tokenizer(refined_summary, return_tensors="pt", max_length=1024, truncation=True)
    refined_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    return bart_tokenizer.decode(refined_ids[0], skip_special_tokens=True)

# Input tweet
input_tweet = "Artificial intelligence in healthcare is revolutionizing patient care, but it poses ethical dilemmas."

# Generate the hierarchical summary
tree_summaries = generate_tree_summary(input_tweet)
final_summary = refine_with_rag(tree_summaries)

print("Tree Summaries:", tree_summaries)
print("Final Summary:", final_summary)
