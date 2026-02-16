import os
import torch
from transformers import BertModel, BertTokenizer
from Bio import SeqIO
import numpy as np

# Load DNABERT tokenizer & model
MODEL_NAME = "zhihan1996/DNA_bert_6"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()

def read_fasta(filepath):
    record = next(SeqIO.parse(filepath, "fasta"))
    return str(record.seq)

def get_kmers(sequence, k=6):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def extract_features_from_folder(folder_path, output_folder, k=6, max_tokens=512, stride=256):
    fasta_files = [f for f in os.listdir(folder_path) if f.endswith(".fasta") or f.endswith(".fa")]

    for fasta_file in fasta_files:
        file_path = os.path.join(folder_path, fasta_file)
        sequence = read_fasta(file_path)
        kmers = get_kmers(sequence, k)
        total_kmers = len(kmers)

        all_embeddings = []

        for start in range(0, total_kmers, stride):
            end = min(start + max_tokens, total_kmers)
            window_kmers = kmers[start:end]
            input_text = " ".join(window_kmers)

            inputs = tokenizer([input_text], return_tensors="pt", padding="max_length", truncation=True, max_length=max_tokens)

            with torch.no_grad():
                outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.squeeze().numpy()
            all_embeddings.append(embeddings)

        # Stack and save
        all_embeddings = np.vstack(all_embeddings)
        out_filename = os.path.splitext(fasta_file)[0] + ".npy"
        out_path = os.path.join(output_folder, out_filename)
        np.save(out_path, all_embeddings)
        print(f"Saved embeddings for {fasta_file} to {out_path}")

# Example usage
input_folder = "C:/Users/DEEPAK/OneDrive/Desktop/Mini10/mutations/benign"
output_folder = "C:/Users/DEEPAK/OneDrive/Desktop/Mini10/Embeddings/benign_embeddings"

os.makedirs(output_folder, exist_ok=True)
extract_features_from_folder(input_folder, output_folder)
