
import json
import math
import re
from pathlib import Path
from collections import Counter

class HybridInferenceEngine:
    def __init__(self, model_json_path):
        self.vocabulary = {}
        self.idf = []
        self.centroids = []
        self.loaded = False
        self._load_model(model_json_path)

    def _load_model(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocabulary = data['vocabulary'] # word -> index
                self.idf = data['idf'] # index -> weight
                self.centroids = data['centroids'] # list of lists
                self.loaded = True
        except Exception as e:
            print(f"Failed to load lightweight model: {e}")
            self.loaded = False

    def predict(self, texts):
        if not self.loaded:
            return []
        
        labels = []
        for text in texts:
            # 1. Vectorize
            vec = self._vectorize(text)
            # 2. Find Closest Centroid
            label, _ = self._predict_one(vec)
            labels.append(label)
        return labels

    def predict_full(self, texts):
        if not self.loaded:
            return [], []
        
        labels = []
        confidences = []
        for text in texts:
            vec = self._vectorize(text)
            label, conf = self._predict_one(vec)
            labels.append(label)
            confidences.append(conf)
        return labels, confidences

    def _tokenize(self, text):
        # Approximate Sklearn TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
        text = text.lower()
        words = re.findall(r"\b\w\w+\b", text)
        return words

    def _generate_ngrams(self, words, n_min=1, n_max=3):
        ngrams = []
        for n in range(n_min, n_max + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngrams.append(ngram)
        return ngrams

    def _vectorize(self, text):
        words = self._tokenize(text)
        grams = self._generate_ngrams(words)
        
        # TF Count
        tf_counts = Counter(grams)
        total_terms = len(grams)
        
        if total_terms == 0:
            return {}

        # Compute TF-IDF Sparse Vector {index: value}
        sparse_vec = {}
        norm_sq = 0.0
        
        for gram, count in tf_counts.items():
            if gram in self.vocabulary:
                idx = self.vocabulary[gram]
                weight = count * self.idf[idx]
                sparse_vec[idx] = weight
                norm_sq += weight * weight
        
        # L2 Normalize
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for k in sparse_vec:
                sparse_vec[k] /= norm
        return sparse_vec

    def _predict_one(self, sparse_vec):
        if not sparse_vec:
            return 0, 0.5 # Default to cluster 0, 50% confidence
            
        return self._predict_optimized(sparse_vec)

    def _predict_optimized(self, sparse_vec):
        # We need centroid norms. Lazy init.
        if not hasattr(self, 'centroid_norms'):
             self.centroid_norms = [sum(c*c for c in ctr) for ctr in self.centroids]
        
        vec_norm_sq = 1.0 # We normalized it
        
        dists = []
        for i, c_norm_sq in enumerate(self.centroid_norms):
            # Dot Product
            dot = 0.0
            center = self.centroids[i]
            for idx, val in sparse_vec.items():
                 if idx < len(center):
                     dot += val * center[idx]
            
            # Squared Euclidean Distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b)
            # Since vectors are normalized (||a||=1), this matches Sklearn if centroids are normalized?
            # Actually Sklearn KMeans centroids are NOT normalized usually.
            dist_sq = vec_norm_sq + c_norm_sq - 2 * dot
            dists.append((dist_sq, i))
            
        # Sort by distance (ascending)
        dists.sort(key=lambda x: x[0])
        
        best_cluster = dists[0][1]
        best_dist = dists[0][0]
        
        if len(dists) > 1:
            second_best = dists[1][0]
            # Confidence Logic: Softmax over Distances (More Stable)
            # P(c|x) = exp(-d_c) / sum(exp(-d_i))
            # d_c is squared distance.
            
            # For numerical stability, subtract min distance
            min_d = best_dist
            
            # Use top 5 distances only to sharpen distribution
            top_k = dists[:5]
            
            try:
                exps = [math.exp(-(d[0] - min_d)) for d in top_k]
                sum_exps = sum(exps)
                confidence = exps[0] / sum_exps
            except OverflowError:
                confidence = 1.0 # Should not happen with subtraction
                
            # Clamp between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 1.0 # Only one cluster exists
            
        return best_cluster, round(confidence, 2)

    def get_top_keywords(self, cluster_id, top_n=10):
        """
        Returns the top N keywords for a given cluster centroid.
        """
        if not self.loaded or cluster_id < 0 or cluster_id >= len(self.centroids):
            return []
            
        # Lazy load reverse vocab (Index -> Word)
        if not hasattr(self, 'index_to_word'):
            self.index_to_word = {v: k for k, v in self.vocabulary.items()}
            
        centroid = self.centroids[cluster_id]
        
        # Get indices sorted by weight (descending)
        # centroid is a list of floats
        top_indices = sorted(range(len(centroid)), key=lambda i: centroid[i], reverse=True)[:top_n]
        
        keywords = []
        for idx in top_indices:
            if centroid[idx] > 0: # Only include positive weights
                word = self.index_to_word.get(idx, "")
                if word:
                    keywords.append(word)
                    
        return keywords
