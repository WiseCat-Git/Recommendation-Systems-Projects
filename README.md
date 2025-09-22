# Recommendation Systems Projects - Master's in Applied Artificial Intelligence

This repository contains comprehensive recommendation system projects completed as part of my Master's program in Applied Artificial Intelligence. The projects demonstrate a complete progression from classical collaborative filtering techniques to state-of-the-art deep learning approaches for recommendation systems.

## Projects Overview

### Unit 1: Classical Collaborative Filtering Systems
**Files:** `solución_caso_práctico_iep_iaa_rs_u1.py`

#### Multi-Algorithm Movie Recommendation System
- **Dataset:** MovieLens movie ratings dataset
- **Objective:** Implement and compare traditional recommendation approaches
- **Algorithms Implemented:**
  - **Popularity-based Recommendations** with IMDB weighted scoring
  - **User-based Collaborative Filtering** with Pearson correlation and cosine similarity
  - **Item-based Collaborative Filtering** with cosine similarity matrix
  - **Hybrid Recommendation System** combining multiple approaches

**Key Features:**
- Robust data preprocessing with automatic column detection
- Advanced quality filters (minimum ratings per movie/user)
- Comprehensive evaluation metrics (Coverage, Novelty, Diversity, Popularity Bias)
- Interactive visualizations for comparative analysis
- Cold start problem handling strategies

**Evaluation Metrics:**
- **Coverage:** Percentage of catalog that can be recommended
- **Novelty:** Average of 1/(number of ratings) - higher = more novel
- **Diversity:** Intra-list diversity using similarity measures
- **Anti-Popularity Bias:** Tendency to recommend less popular items

**Results Highlights:**
- Hybrid system effectively combines strengths of different approaches
- User-based CF excels at personalization for active users
- Item-based CF provides semantic coherence and stability
- Comprehensive quality filters improve recommendation relevance

---

### Unit 2: Matrix Factorization and Optimization
**Files:** `solución_caso_práctico_u2_iep_iaa_rs.py`

#### Advanced Matrix Factorization with Hyperparameter Optimization
- **Objective:** Implement matrix factorization from scratch with comprehensive optimization
- **Models Implemented:**
  - **TruncatedSVD Recommender** using Scikit-learn
  - **Custom SGD Matrix Factorization** with full control over optimization
  - **Grid Search Optimization** for hyperparameter tuning
  - **Random Search Optimization** as efficient alternative

**Technical Implementation:**
- **Custom SGD Algorithm:** Complete implementation with L2 regularization
- **Bias Modeling:** Global, user-specific, and item-specific biases
- **Memory Optimization:** Sparse matrix representations using SciPy
- **Efficient Data Structures:** Optimized user-item mappings and indexing

**Hyperparameter Optimization:**
- **Grid Search:** Exhaustive exploration of parameter space
- **Random Search:** Efficient alternative for large parameter spaces
- **Parameters Tuned:** Number of factors, learning rate, regularization, epochs

**Performance Results:**

| Model | RMSE | MAE | Improvement vs Baseline |
|-------|------|-----|-------------------------|
| TruncatedSVD | 1.024 | 0.812 | Baseline |
| SGD Baseline | 0.987 | 0.789 | +3.6% |
| SGD Grid Search | 0.952 | 0.756 | +7.0% |
| SGD Random Search | 0.943 | 0.751 | +7.9% |

**Key Technical Contributions:**
- Mathematical foundation: R ≈ μ + b_u + b_i + p_u^T q_i
- Xavier/Glorot initialization for stable convergence
- Comprehensive loss function with regularization
- Production-ready code with extensive error handling

---

### Unit 3: Deep Learning with TensorFlow Recommenders
**Files:** `solución_enunciado_caso_práctico_recommendation_systems_u3_iep_iaa.py`

#### Neural Collaborative Filtering with TensorFlow
- **Dataset:** MovieLens 100K via TensorFlow Datasets
- **Objective:** Implement state-of-the-art deep learning recommendation system
- **Framework:** TensorFlow Recommenders (TFR) with custom adaptations

**Advanced Architecture:**
- **Embedding Layers:** Dense vector representations for users and items
- **Neural Network:** Multi-layer perceptron for non-linear interactions
- **Custom Model Class:** Full control over forward pass and training loop
- **Feature Engineering:** Advanced data preparation and vocabulary management

**Technical Challenges Solved:**
- **API Compatibility:** Resolved TensorFlow 2.19+ compatibility issues
- **Tensor Format Alignment:** Fixed Keras training loop integration
- **Vocabulary Corruption:** Robust handling of mixed data types
- **Cold Start Handling:** Fallback strategies for unknown users/items

**Model Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| RMSE | 0.0616 | Root Mean Square Error |
| MAE | 0.0284 | Mean Absolute Error |
| R² | 0.5491 | Coefficient of Determination |
| Precision@10 | 0.597 | Top-10 Accuracy |

**Critical Engineering Solutions:**
```python
# Key architectural fix for Keras compatibility
def call(self, inputs):
    return rating  # Return tensor directly, not dictionary

# Robust data preprocessing
def clean_vocab_item(self, item):
    # Handle bytes, strings, null values consistently
    # Filter problematic characters
    return cleaned_item
```

**Advanced Features:**
- **Batch Processing:** Efficient DataLoader implementation with PyTorch-style batching
- **Model Serialization:** Save/load functionality for production deployment
- **Evaluation Pipeline:** Comprehensive metrics including AUC and Precision@K
- **Recommendation Generation:** Real-time inference with similarity-based retrieval

---

## Technical Skills Demonstrated

### Programming & Frameworks
- **Python:** Advanced NumPy, Pandas operations with memory optimization
- **TensorFlow:** Neural network architecture, custom training loops, TFR integration
- **Scikit-learn:** Matrix decomposition, model evaluation, hyperparameter optimization
- **SciPy:** Sparse matrix operations, statistical computations

### Machine Learning Techniques
- **Collaborative Filtering:** User-based and item-based approaches
- **Matrix Factorization:** SVD, SGD-based optimization, bias modeling
- **Deep Learning:** Neural collaborative filtering, embedding layers
- **Optimization:** Grid search, random search, gradient descent

### Data Engineering
- **Data Preprocessing:** Robust cleaning, type handling, vocabulary management
- **Feature Engineering:** User/item embeddings, temporal features
- **Evaluation Framework:** Comprehensive metrics, cross-validation
- **Performance Optimization:** Memory-efficient sparse representations

### Advanced Concepts
- **Cold Start Problem:** Multiple fallback strategies
- **Bias Modeling:** Global, user, and item-specific biases
- **Regularization:** L2 penalty, dropout, early stopping
- **Similarity Metrics:** Cosine, Pearson correlation, neural similarity

---

## Evaluation Framework

### Classical Metrics
- **Accuracy:** RMSE, MAE for rating prediction
- **Ranking:** Precision@K, Recall@K for recommendation lists
- **Diversity:** Intra-list diversity, catalog coverage
- **Novelty:** Popularity bias, serendipity measures

### Advanced Analysis
- **Statistical Significance:** Pearson correlation p-values
- **Convergence Analysis:** Training loss curves, learning rate optimization
- **Ablation Studies:** Component-wise performance analysis
- **Scalability Testing:** Performance with varying dataset sizes

---

## Repository Contents

```
├── solución_caso_práctico_iep_iaa_rs_u1.py    # Unit 1: Classical CF Systems
├── solución_caso_práctico_u2_iep_iaa_rs.py    # Unit 2: Matrix Factorization
├── solución_enunciado_caso_práctico_rs_u3.py  # Unit 3: Deep Learning
└── README.md                                   # This file
```

---

## Getting Started

To run these projects:

1. Clone the repository
2. Install required dependencies:
   ```bash
   # Classical approaches (Units 1-2)
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   
   # Deep learning (Unit 3)
   pip install tensorflow tensorflow-recommenders torch
   
   # Data processing and evaluation
   pip install urllib3 zipfile
   ```
3. Run individual Python files in your preferred environment

**Note:** Unit 3 requires significant computational resources for neural network training.

---

## Project Evolution and Learning Path

### Unit 1: Foundation Building
- **Focus:** Understanding collaborative filtering fundamentals
- **Learning:** User behavior modeling, similarity metrics, evaluation
- **Output:** Multi-algorithm comparison framework

### Unit 2: Mathematical Depth
- **Focus:** Matrix factorization theory and implementation
- **Learning:** Optimization algorithms, regularization, hyperparameter tuning
- **Output:** Production-ready SGD implementation

### Unit 3: Modern Approaches
- **Focus:** Neural networks for recommendation systems
- **Learning:** Deep learning architectures, TensorFlow ecosystem
- **Output:** Scalable neural collaborative filtering system

---

## Key Insights and Contributions

### Algorithmic Insights
1. **Hybrid approaches consistently outperform single methods**
2. **Proper bias modeling is crucial for matrix factorization accuracy**
3. **Neural networks excel at capturing non-linear user-item interactions**
4. **Hyperparameter optimization provides significant performance gains**

### Engineering Contributions
1. **Robust data preprocessing** handling real-world data inconsistencies
2. **Memory-efficient implementations** using sparse matrices and optimized indexing
3. **Comprehensive evaluation framework** with multiple quality metrics
4. **Production-ready code** with extensive error handling and documentation

### Research Applications
1. **Cold start strategies** for new users and items
2. **Bias analysis** in recommendation systems
3. **Scalability considerations** for large-scale deployments
4. **Evaluation methodology** for offline and online testing

---

## Performance Benchmarks

### Classical Methods (Unit 1)
- **Popularity:** High coverage, low personalization
- **User-CF:** High personalization, requires active users
- **Item-CF:** Stable performance, semantic coherence
- **Hybrid:** Best overall balance

### Matrix Factorization (Unit 2)
- **TruncatedSVD:** Fast, good baseline performance
- **Custom SGD:** Superior accuracy with proper tuning
- **Optimization:** 7.9% improvement with random search

### Deep Learning (Unit 3)
- **Neural CF:** Best accuracy on complex patterns
- **Embedding Quality:** Rich user/item representations
- **Scalability:** Handles large vocabularies efficiently

---

## Business Applications

### E-commerce
- Product recommendation engines
- Cross-selling and upselling strategies
- Customer segmentation and targeting

### Media & Entertainment
- Content recommendation (movies, music, articles)
- Personalized playlists and queues
- Content discovery and exploration

### Social Networks
- Friend and connection recommendations
- Content feed personalization
- Community and group suggestions

---

## Academic Context

These projects were completed as part of the Master's program in Applied Artificial Intelligence, specifically within the Recommendation Systems track. The progression demonstrates:

**Theoretical Foundation:** Deep understanding of collaborative filtering mathematics and optimization theory

**Practical Implementation:** Ability to translate algorithms into efficient, scalable code

**Modern Techniques:** Proficiency with state-of-the-art deep learning frameworks

**Evaluation Rigor:** Comprehensive testing methodology with multiple quality metrics

The projects bridge academic research and industry applications, providing both theoretical depth and practical utility for real-world recommendation system deployment.

---

## Future Enhancements

### Technical Improvements
- **Multi-armed Bandit** approaches for online learning
- **Graph Neural Networks** for social recommendation
- **Transformer architectures** for sequential recommendation
- **Federated Learning** for privacy-preserving recommendations

### Evaluation Extensions
- **A/B Testing** framework for online evaluation
- **Causal Inference** methods for unbiased evaluation
- **Long-term Impact** metrics beyond immediate accuracy
- **Fairness and Bias** comprehensive analysis

---

## License

This project is part of academic coursework for educational purposes.

---

## Contact

For questions about these projects or collaboration opportunities, please feel free to reach out through GitHub.
