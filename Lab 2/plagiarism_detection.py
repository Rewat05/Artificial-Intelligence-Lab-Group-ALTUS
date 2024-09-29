import re
import heapq

def preprocess_text(text):
    """Preprocess the text: normalize, tokenize, and remove stopwords."""
    # Normalize text: lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize into sentences
    sentences = text.split('.')
    processed_sentences = []
    
    for sentence in sentences:
        # Remove stopwords and keep only alphabetic words
        words = [word for word in sentence.split() if word.isalpha()]
        processed_sentences.append(' '.join(words))
    
    return processed_sentences

def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def heuristic(i, j, len1, len2):
    """Heuristic function for A* search."""
    return abs(len1 - i) + abs(len2 - j)

def a_star_search(sentences1, sentences2):
    """Align sentences from two documents using the A* search algorithm."""
    len1, len2 = len(sentences1), len(sentences2)
    
    open_set = [(0 + heuristic(0, 0, len1, len2), 0, 0, 0)]  # (f_score, g_score, i, j)
    heapq.heapify(open_set)
    g_score = {(0, 0): 0}
    came_from = {}
    
    while open_set:
        _, cost, i, j = heapq.heappop(open_set)
        
        if i == len1 and j == len2:
            break
        
        for next_i, next_j in [(i+1, j), (i, j+1), (i+1, j+1)]:
            if next_i <= len1 and next_j <= len2:
                if next_i < len1 and next_j < len2:
                    additional_cost = levenshtein_distance(sentences1[i], sentences2[j])
                else:
                    additional_cost = 0  # No cost if skipping past end
                
                new_cost = cost + additional_cost
                if (next_i, next_j) not in g_score or new_cost < g_score[(next_i, next_j)]:
                    g_score[(next_i, next_j)] = new_cost
                    f_score = new_cost + heuristic(next_i, next_j, len1, len2)
                    heapq.heappush(open_set, (f_score, new_cost, next_i, next_j))
                    came_from[(next_i, next_j)] = (i, j)
    
    alignments = []
    current = (len1, len2)
    while current in came_from:
        prev = came_from[current]
        if prev[0] < len1 and prev[1] < len2:
            alignments.append((sentences1[prev[0]], sentences2[prev[1]]))
        current = prev
    alignments.reverse()
    
    return alignments

def calculate_similarity(s1, s2):
    """Calculate the similarity between two sentences based on Levenshtein distance."""
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:  # Avoid division by zero
        return 1.0
    return 1 - (dist / max_len)

def detect_plagiarism(sentences1, sentences2):
    """Detect potential plagiarism and return all similar sentence pairs."""
    alignments = a_star_search(sentences1, sentences2)
    plagiarized_pairs = []
    
    for sent1, sent2 in alignments:
        similarity = calculate_similarity(sent1, sent2)
        plagiarized_pairs.append((sent1, sent2, similarity))
    
    return plagiarized_pairs

def calculate_plagiarism_level(plagiarized_pairs):
    """Calculate the overall level of plagiarism."""
    if not plagiarized_pairs:
        return 0.0  # No plagiarism detected
    
    total_similarity = sum(similarity for _, _, similarity in plagiarized_pairs)
    num_pairs = len(plagiarized_pairs)
    return total_similarity / num_pairs  # Average similarity score

# New example usage with different text inputs
text1 = "Machine learning is a subset of artificial intelligence, focusing on building systems that learn from data and improve their performance over time. It's a key technology in data-driven industries, providing insights and automation. Popular applications include recommendation systems, image recognition, and predictive analytics."
text2 = "Machine learning is a branch of artificial intelligence. It involves creating models that learn from data and get better with experience. This technology powers applications like recommendation systems, image classification, and predictive analytics."

# Preprocess and tokenize the documents
processed_sent1 = preprocess_text(text1)
processed_sent2 = preprocess_text(text2)

print("Processed Sentences Doc1:", processed_sent1)
print("Processed Sentences Doc2:", processed_sent2)

# Detect plagiarism based on similarity
plagiarized_pairs = detect_plagiarism(processed_sent1, processed_sent2)
print("Plagiarized Pairs:")
for sent1, sent2, similarity in plagiarized_pairs:
    print(f"'{sent1}' is similar to '{sent2}' with similarity score of {similarity:.2f}")

# Calculate the overall level of plagiarism
plagiarism_level = calculate_plagiarism_level(plagiarized_pairs)
print(f"Overall level of plagiarism: {plagiarism_level:.2f}")