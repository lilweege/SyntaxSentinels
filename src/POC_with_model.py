from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
import torch
import os
import ast
import torch
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm  # Import tqdm for progress bar
import json
import math
import ast
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CodeBERT tokenizer and model for embeddings
# huggingface_model_name = "microsoft/codebert-base"
# tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
# model = AutoModel.from_pretrained(huggingface_model_name).to(device)

huggingface_model_name = "FacebookAI/roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(huggingface_model_name)
model = RobertaModel.from_pretrained(huggingface_model_name)


def token_similarity(code1, code2):
    """Calculate token-based similarity using CountVectorizer and cosine similarity."""
    # Tokenization and vectorization
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([code1, code2]).toarray()

    # Convert vectors to tensors and ensure they're float type for cosine similarity calculation
    tensor1 = torch.tensor(vectors[0], dtype=torch.float32)
    tensor2 = torch.tensor(vectors[1], dtype=torch.float32)

    # Compute cosine similarity
    cosine_similarity = (tensor1 @ tensor2) / (torch.norm(tensor1, p=2) * torch.norm(tensor2, p=2))
    return cosine_similarity.item()  # Convert to Python float for easier handling


def ast_similarity(code1, code2):
    # Dictionary for Python AST node types (example; can be extended as needed)
    nodetypedict = {node: i for i, node in enumerate(ast.__dict__.keys())}

    def create_adjacency_matrix(ast_tree):
        """Generate an adjacency matrix from an AST tree."""
        matrix_size = len(nodetypedict)
        matrix = np.zeros((matrix_size, matrix_size))
        
        def traverse(node, parent=None):
            """Recursive traversal of the AST tree."""
            if not isinstance(node, ast.AST):
                return
            
            current_type = nodetypedict.get(type(node).__name__, -1)
            parent_type = nodetypedict.get(type(parent).__name__, -1) if parent else -1

            if parent is not None and current_type >= 0 and parent_type >= 0:
                matrix[parent_type][current_type] += 1

            for child in ast.iter_child_nodes(node):
                traverse(child, parent=node)

        traverse(ast_tree)
        # Normalize the matrix
        for row in range(matrix.shape[0]):
            total = matrix[row].sum()
            if total > 0:
                matrix[row] /= total
        return matrix

    def compute_similarity(matrix1, matrix2):
        """Compute similarity between two matrices using cosine similarity."""
        vec1 = matrix1.flatten().reshape(1, -1)
        vec2 = matrix2.flatten().reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity

    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    matrix1 = create_adjacency_matrix(tree1)
    matrix2 = create_adjacency_matrix(tree2)
    return compute_similarity(matrix1, matrix2)


def process_code_pairs_with_progress(file_pairs):
    """Process file pairs and calculate combined similarity scores with progress bar."""
    results = []
    for (file1, code1), (file2, code2) in tqdm(file_pairs, desc="Processing code pairs", unit="pair"):
        token_sim, ast_sim, embed_sim = all_similarity(code1, code2)

        results.append({
            "file1": file1,
            "file2": file2,
            "embed_sim": embed_sim,
            "token_sim": token_sim,
            "ast_sim": ast_sim,
        })
    return results


@cache
def get_code_embedding(code_snippet):
    """Generate an embedding for a code snippet using CodeBERT."""
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token representation as the embedding
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

def embedding_similarity(code1, code2):
    """Calculate cosine similarity between embeddings of two code snippets."""
    embedding1 = get_code_embedding(code1)
    embedding2 = get_code_embedding(code2)
    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cosine_sim.item()

# Example of integrating embedding similarity with existing functions
def all_similarity(code1, code2):
    # Calculate existing similarities
    token_sim = token_similarity(code1, code2)
    ast_sim = ast_similarity(code1, code2)

    # Calculate embedding-based similarity
    embed_sim = embedding_similarity(code1, code2)
    # Affine
    normalized_embed_sim = max(0.0, min(1.0, (embed_sim - 0.99) * 100))
    # Sigmoid
    normalized_embed_sim = 1 / (1 + math.exp(-9*(normalized_embed_sim-0.5)))

    return token_sim, ast_sim, normalized_embed_sim


extraction_path = "Project_CodeNet_Python800"
sample_path = "p02618_3_small"

# sample_files = os.listdir(os.path.join(extraction_path, sample_path))
cluster1 = ["cluster1_A.py", "cluster1_B.py"]
cluster2 = ["cluster2_A.py", "cluster2_B.py", "cluster2_C.py"]
sample_files = [*cluster1, *cluster2]

file_pairs = []
for i in range(len(sample_files)):
    for j in range(i + 1, len(sample_files)):
        file1_path = os.path.join(extraction_path, sample_path, sample_files[i])
        file2_path = os.path.join(extraction_path, sample_path, sample_files[j])

        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            code1 = f1.read()
            code2 = f2.read()

        file_pairs.append(((sample_files[i], code1), (sample_files[j], code2)))

# Process file pairs and calculate similarities
labeled_data_with_progress = process_code_pairs_with_progress(file_pairs)

# Sort and display top results
labeled_data_with_progress.sort(key=lambda x: x["embed_sim"], reverse=True)

# Save the results to a file
with open("similarity_results.json", "w") as f:
    json.dump(labeled_data_with_progress, f, indent=4)
