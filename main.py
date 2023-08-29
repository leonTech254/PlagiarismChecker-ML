
"""
Author:Leon Martin
Email:martinleontech23@gmail.com
WhatsApp:+254719531573
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_and_preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
       
        processed_text = text.lower()  
        return processed_text

def get_txt_files_in_directory(directory):
    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]
    return txt_files

def calculate_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_matrix[0][0]
txt_directory = "./files"
txt_files = get_txt_files_in_directory(txt_directory)
for i in range(len(txt_files)):
    for j in range(i + 1, len(txt_files)):
        file1_path = os.path.join(txt_directory, txt_files[i])
        file2_path = os.path.join(txt_directory, txt_files[j])
        
        text1 = read_and_preprocess_text(file1_path)
        text2 = read_and_preprocess_text(file2_path)
        
        similarity = calculate_similarity(text1, text2)
        
        print(f"Similarity between {txt_files[i]} and {txt_files[j]}: {similarity:.2f}")
