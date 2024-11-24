import os
import ast
import torch
import difflib
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm  # Import tqdm for progress bar

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def is_python3_file(file_path):
    """Check if a file is Python 3 compatible by attempting to parse it."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        ast.parse(code)  # Will raise an error if not valid Python 3 code
        return True
    except SyntaxError:
        return False


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
    """Calculate AST-based similarity between two code snippets."""
    try:
        tree1 = ast.dump(ast.parse(code1))
        tree2 = ast.dump(ast.parse(code2))
    except SyntaxError:
        return 0.0  # Return 0 if there's a syntax error (invalid code)

    # Tokenizing the AST output
    return token_similarity(tree1, tree2)


def hybrid_similarity(code1, code2, thresholds):
    """Hybrid similarity: combines token and AST-based similarity."""
    token_thresh, ast_thresh = thresholds

    # Calculate token and AST similarity
    token_sim = token_similarity(code1, code2)
    ast_sim = ast_similarity(code1, code2)

    # Weigh the similarities based on the thresholds
    hybrid_score = (token_sim * token_thresh) + (ast_sim * ast_thresh)
    return hybrid_score


def process_code_pairs(file_pairs, thresholds):
    """Process file pairs and calculate hybrid similarity scores."""
    labeled_data = []
    for (file1, code1), (file2, code2) in tqdm(file_pairs, desc="Processing code pairs", unit="pair"):
        if not is_python3_file(file1) or not is_python3_file(file2):
            continue  # Skip non-Python3 files
        
        similarity_score = hybrid_similarity(code1, code2, thresholds)
        labeled_data.append({
            "file1": file1,
            "file2": file2,
            "similarity_score": similarity_score
        })
    
    return labeled_data


def read_code_files_for_question(directory, question_folder):
    """Read Python files from a specific question folder."""
    file_pairs = []
    question_path = os.path.join(directory, question_folder)
    if os.path.isdir(question_path):
        python_files = [f for f in os.listdir(question_path) if f.endswith('.py') and is_python3_file(os.path.join(question_path, f))]
        for i, file1 in enumerate(python_files):
            for file2 in python_files[i + 1:]:
                file1_path = os.path.join(question_path, file1)
                file2_path = os.path.join(question_path, file2)
                
                # Read the code from both files
                with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
                    code1 = f1.read()
                    code2 = f2.read()
                
                file_pairs.append(((file1_path, code1), (file2_path, code2)))
    
    return file_pairs


def save_similarity_scores(labeled_data, question_name, output_path):
    """Save similarity scores to a text file."""
    output_file = os.path.join(output_path, f"{question_name}_similarity_scores.txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        for data in labeled_data:
            file.write(f"{data['file1']} | {data['file2']} | Similarity: {data['similarity_score']:.4f}\n")
    print(f"Similarity scores saved to: {output_file}")


# Example Usage

# Directory where Python code files are located
directory = "Project_CodeNet_Python800"
question_folder = "p00000"  # Replace with the specific question folder you want to process

# Define your similarity thresholds
thresholds = (0.3, 0.6)  # Token threshold = 0.3, AST threshold = 0.6

# Read code files and generate pairs for the specific question folder
file_pairs = read_code_files_for_question(directory, question_folder)

# Process file pairs and get similarity labels
labeled_data = process_code_pairs(file_pairs, thresholds)

# sort labeled_data by similarity_score from highest to lowest
labeled_data.sort(key=lambda x: x["similarity_score"], reverse=True)

# Save results
output_path = "similarity_scores"        # Folder to save results
save_similarity_scores(labeled_data, question_folder, output_path)