"""
model.py — AI Module for SkillForge
=====================================
This module implements the core AI/ML pipeline for teammate recommendation
and compatibility prediction. It uses scikit-learn for all ML operations.

AI Modules Implemented:
    1. Skill Vectorization (TF-IDF)
    2. Similarity Engine (Cosine Similarity)
    3. Feature Engineering
    4. Compatibility Prediction (Logistic Regression)
    5. Team Balance Analyzer

Technical Explanation:
----------------------

TF-IDF (Term Frequency - Inverse Document Frequency):
    TF-IDF is a numerical statistic that reflects how important a word (skill)
    is to a document (user profile) in a collection (all users).
    
    - Term Frequency (TF): How often a skill appears in a user's profile.
      TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)
    
    - Inverse Document Frequency (IDF): How rare/unique a skill is across all users.
      IDF(t) = log(Total number of documents / Number of documents containing term t)
    
    - TF-IDF(t,d) = TF(t,d) × IDF(t)
    
    Skills that are common across all users (like "python") get lower weight,
    while unique skills (like "computer vision") get higher weight.

Cosine Similarity:
    Measures the cosine of the angle between two TF-IDF vectors.
    
    cos(θ) = (A · B) / (||A|| × ||B||)
    
    - Value ranges from 0 (completely different) to 1 (identical).
    - It measures orientation, not magnitude, making it ideal for text similarity.
    - Two users with similar skill sets will have vectors pointing in similar directions.

Logistic Regression:
    A supervised ML algorithm for binary classification.
    
    P(compatible) = 1 / (1 + e^(-z))
    where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    
    - Uses sigmoid function to output probability between 0 and 1.
    - We use engineered features (skill similarity, experience difference, etc.)
      as input features.
    - Output: Probability of compatibility (0-100%).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MODULE 1: Skill Vectorization using TF-IDF
# ============================================================================

class SkillVectorizer:
    """
    Converts user skill text into numerical vectors using TF-IDF.
    
    Why TF-IDF?
        - It captures the importance of each skill relative to the entire user base.
        - Common skills are down-weighted; rare/specialized skills are up-weighted.
        - This produces meaningful numerical representations for similarity computation.
    
    Example:
        User A: "python, react, tensorflow"
        User B: "python, java, spring"
        
        'python' is common → lower TF-IDF weight
        'tensorflow' is rare → higher TF-IDF weight
        
        Result: User A's vector emphasizes ML skills more than generic programming.
    """

    def __init__(self):
        # Initialize TF-IDF Vectorizer with custom settings for skill text
        self.vectorizer = TfidfVectorizer(
            # Don't convert to lowercase (we pre-process skills)
            lowercase=True,
            # Use both unigrams and bigrams to capture multi-word skills
            # e.g., "machine learning" as a single feature
            ngram_range=(1, 2),
            # Custom tokenizer that splits on commas (skills are comma-separated)
            tokenizer=lambda x: [s.strip() for s in x.split(',') if s.strip()],
            # Don't apply stop word removal (skill names are meaningful)
            stop_words=None,
            # Sublinear TF scaling: use 1 + log(tf) instead of raw tf
            # This prevents very long skill lists from dominating
            sublinear_tf=True
        )
        self.is_fitted = False

    def fit_transform(self, skill_texts):
        """
        Fit the vectorizer on all user skill texts and transform them.
        
        Args:
            skill_texts (list[str]): List of comma-separated skill strings
        
        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
                Shape: (n_users, n_unique_skills)
        """
        vectors = self.vectorizer.fit_transform(skill_texts)
        self.is_fitted = True
        return vectors

    def transform(self, skill_texts):
        """
        Transform new skill texts using the already-fitted vectorizer.
        
        Args:
            skill_texts (list[str]): New skill strings to vectorize
        
        Returns:
            scipy.sparse.csr_matrix: TF-IDF vectors for new users
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first. Call fit_transform().")
        return self.vectorizer.transform(skill_texts)

    def get_feature_names(self):
        """Return the list of skills/features learned by the vectorizer."""
        return self.vectorizer.get_feature_names_out().tolist()


# ============================================================================
# MODULE 2: Similarity Engine using Cosine Similarity
# ============================================================================

class SimilarityEngine:
    """
    Computes similarity between users based on their skill vectors.
    
    How Cosine Similarity Works:
        1. Each user's skills are represented as a vector in high-dimensional space.
        2. Cosine similarity measures the angle between two vectors.
        3. If two users share similar skills, their vectors point in similar directions.
        4. cos(0°) = 1.0 means identical skill profiles.
        5. cos(90°) = 0.0 means completely different skill profiles.
    
    Why Cosine Similarity (not Euclidean distance)?
        - It's independent of vector magnitude (skill list length).
        - A user with 3 skills and another with 10 skills can still be compared fairly.
        - It focuses on the pattern of skills, not the quantity.
    """

    @staticmethod
    def compute_similarity(vector_a, vector_b):
        """
        Compute cosine similarity between two skill vectors.
        
        Args:
            vector_a: TF-IDF vector of user A
            vector_b: TF-IDF vector of user B
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        similarity = cosine_similarity(vector_a, vector_b)
        return float(similarity[0][0])

    @staticmethod
    def find_top_similar(target_vector, all_vectors, user_ids, top_n=3, exclude_id=None):
        """
        Find the top N most similar users to the target user.
        
        Algorithm:
            1. Compute cosine similarity between target and every other user.
            2. Sort by similarity score (descending).
            3. Return top N results (excluding the target user if needed).
        
        Args:
            target_vector: TF-IDF vector of the target user
            all_vectors: TF-IDF matrix of all users
            user_ids: List of user IDs corresponding to each row in all_vectors
            top_n (int): Number of top similar users to return
            exclude_id: User ID to exclude from results (the target user)
        
        Returns:
            list[tuple]: List of (user_id, similarity_score) tuples
        """
        # Compute similarity between target and all users at once
        similarities = cosine_similarity(target_vector, all_vectors).flatten()

        # Create (user_id, score) pairs and sort by score descending
        scored_users = list(zip(user_ids, similarities))

        # Exclude the target user from results
        if exclude_id is not None:
            scored_users = [(uid, score) for uid, score in scored_users if uid != exclude_id]

        # Sort by similarity score (highest first)
        scored_users.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return scored_users[:top_n]


# ============================================================================
# MODULE 3: Feature Engineering
# ============================================================================

class FeatureEngineer:
    """
    Creates structured features from raw user data for the ML model.
    
    Feature Engineering Explanation:
        Raw data (skills, experience, domain) cannot be directly fed into ML models.
        We need to transform them into meaningful numerical features.
    
    Engineered Features:
        1. skill_similarity (float, 0-1):
           - The cosine similarity score between two users' skill vectors.
           - Higher = more similar skills.
        
        2. experience_difference (int, 0-2):
           - Absolute difference in experience levels.
           - beginner=0, intermediate=1, advanced=2
           - Lower difference = more compatible experience levels.
        
        3. domain_match (binary, 0 or 1):
           - Whether two users are from the same domain.
           - 1 = same domain (complementary within same field).
           - 0 = different domains.
        
        4. interest_match (binary, 0 or 1):
           - Whether two users share the same hackathon interest.
           - 1 = aligned interests (want to work on same type of project).
           - 0 = different interests.
    
    Why These Features?
        - skill_similarity: Core metric — similar skills suggest similar technical language.
        - experience_difference: Teams with mixed experience levels often perform well.
        - domain_match: People in the same domain understand each other's challenges.
        - interest_match: Shared hackathon interests = higher motivation to collaborate.
    """

    # Experience level encoding (ordinal)
    EXPERIENCE_LEVELS = {
        'beginner': 0,
        'intermediate': 1,
        'advanced': 2
    }

    @staticmethod
    def encode_experience(experience):
        """
        Convert experience string to numerical value.
        
        Args:
            experience (str): 'beginner', 'intermediate', or 'advanced'
        
        Returns:
            int: Encoded value (0, 1, or 2)
        """
        return FeatureEngineer.EXPERIENCE_LEVELS.get(experience.lower(), 0)

    @staticmethod
    def compute_features(user_a, user_b, similarity_score):
        """
        Compute the feature vector for a pair of users.
        
        This creates the input features for the Logistic Regression model.
        
        Args:
            user_a (dict): First user's profile data
            user_b (dict): Second user's profile data
            similarity_score (float): Pre-computed cosine similarity
        
        Returns:
            numpy.ndarray: Feature vector [skill_sim, exp_diff, domain_match, interest_match]
        """
        # Feature 1: Skill Similarity Score (continuous, 0.0 to 1.0)
        skill_sim = similarity_score

        # Feature 2: Experience Difference (ordinal, 0 to 2)
        exp_a = FeatureEngineer.encode_experience(user_a.get('experience', 'beginner'))
        exp_b = FeatureEngineer.encode_experience(user_b.get('experience', 'beginner'))
        exp_diff = abs(exp_a - exp_b)

        # Feature 3: Domain Match (binary, 0 or 1)
        domain_match = 1 if user_a.get('domain', '').lower() == user_b.get('domain', '').lower() else 0

        # Feature 4: Hackathon Interest Match (binary, 0 or 1)
        interest_match = 1 if user_a.get('interest', '').lower() == user_b.get('interest', '').lower() else 0

        return np.array([skill_sim, exp_diff, domain_match, interest_match])

    @staticmethod
    def compute_features_batch(user_a, users_b, similarity_scores):
        """
        Compute feature vectors for multiple user pairs at once.
        
        Args:
            user_a (dict): The target user
            users_b (list[dict]): List of candidate users
            similarity_scores (list[float]): Corresponding similarity scores
        
        Returns:
            numpy.ndarray: Feature matrix of shape (n_users, 4)
        """
        features = []
        for user_b, sim_score in zip(users_b, similarity_scores):
            feat = FeatureEngineer.compute_features(user_a, user_b, sim_score)
            features.append(feat)
        return np.array(features)


# ============================================================================
# MODULE 4: Compatibility Prediction using Logistic Regression
# ============================================================================

class CompatibilityPredictor:
    """
    Predicts compatibility probability between two users using Logistic Regression.
    
    Logistic Regression Explanation:
        Logistic Regression is a supervised classification algorithm that predicts
        the probability of a binary outcome (compatible / not compatible).
        
        Mathematical Model:
            1. Linear combination: z = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
               where x₁ = skill_similarity, x₂ = experience_diff,
                     x₃ = domain_match, x₄ = interest_match
            
            2. Sigmoid activation: P(compatible) = 1 / (1 + e^(-z))
               This maps the linear output to a probability between 0 and 1.
            
            3. Decision: If P > 0.5, predict "compatible", else "not compatible".
        
        Training Process:
            - We generate synthetic training data based on known compatibility rules.
            - The model learns the weights (w₁, w₂, w₃, w₄) that best predict compatibility.
            - After training, it can predict compatibility for any new user pair.
        
        Why Logistic Regression?
            - Simple, interpretable, and effective for binary classification.
            - Works well with engineered features.
            - Provides probability output (not just yes/no), perfect for scoring.
            - Suitable for academic demonstration of ML concepts.
    """

    def __init__(self):
        self.model = LogisticRegression(
            # L2 regularization to prevent overfitting
            penalty='l2',
            # Regularization strength (lower = more regularization)
            C=1.0,
            # Use LBFGS solver (efficient for small datasets)
            solver='lbfgs',
            # Maximum iterations for convergence
            max_iter=1000,
            # Random seed for reproducibility
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def generate_training_data(self, n_samples=500):
        """
        Generate synthetic training data for the compatibility model.
        
        Why Synthetic Data?
            In a real-world scenario, we would collect actual compatibility feedback
            from hackathon participants. For this academic project, we generate
            realistic synthetic data based on domain knowledge about what makes
            good hackathon teams.
        
        Compatibility Rules (used to generate labels):
            - High skill similarity + same interest → HIGH compatibility
            - Mixed experience levels + domain match → MODERATE-HIGH compatibility  
            - Low skill similarity + different interests → LOW compatibility
            - Same domain + high similarity → HIGH compatibility
        
        Args:
            n_samples (int): Number of training samples to generate
        
        Returns:
            tuple: (X_train, y_train) where X is features and y is labels
        """
        np.random.seed(42)  # Reproducibility

        X = []
        y = []

        for _ in range(n_samples):
            # Generate random feature values
            skill_sim = np.random.uniform(0, 1)
            exp_diff = np.random.choice([0, 1, 2])
            domain_match = np.random.choice([0, 1])
            interest_match = np.random.choice([0, 1])

            features = [skill_sim, exp_diff, domain_match, interest_match]

            # Define compatibility based on realistic rules
            # This encodes domain knowledge about hackathon team dynamics
            compatibility_score = (
                0.35 * skill_sim +              # Skill similarity is most important
                0.20 * (1 - exp_diff / 2) +     # Smaller experience gap is better
                0.25 * domain_match +            # Same domain helps collaboration
                0.20 * interest_match            # Shared interest boosts motivation
            )

            # Add some noise for realistic variation
            compatibility_score += np.random.normal(0, 0.08)
            compatibility_score = np.clip(compatibility_score, 0, 1)

            # Binary label: compatible if score > 0.45 threshold
            label = 1 if compatibility_score > 0.45 else 0

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)

    def train(self):
        """
        Train the Logistic Regression model on synthetic data.
        
        Training Pipeline:
            1. Generate synthetic training data (feature vectors + labels).
            2. Scale features using StandardScaler (zero mean, unit variance).
            3. Fit Logistic Regression model on scaled features.
            4. Mark model as trained.
        
        Feature Scaling (StandardScaler):
            - Transforms each feature to have mean=0 and std=1.
            - Essential because features have different scales:
              skill_similarity: 0-1, experience_diff: 0-2
            - Without scaling, features with larger values would dominate.
        """
        X_train, y_train = self.generate_training_data(n_samples=500)

        # Scale features for better model performance
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train the logistic regression model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Log training info
        accuracy = self.model.score(X_train_scaled, y_train)
        print(f"[AI MODEL] Logistic Regression trained. Training accuracy: {accuracy:.2%}")
        print(f"[AI MODEL] Feature weights: {self.model.coef_[0]}")
        print(f"[AI MODEL] Intercept: {self.model.intercept_[0]:.4f}")

    def predict_compatibility(self, features):
        """
        Predict compatibility probability for a user pair.
        
        Args:
            features (numpy.ndarray): Feature vector [skill_sim, exp_diff, domain_match, interest_match]
        
        Returns:
            float: Compatibility probability (0-100%)
        
        How Prediction Works:
            1. Scale the input features using the same scaler used during training.
            2. Feed scaled features to the trained Logistic Regression model.
            3. Get probability of class 1 (compatible).
            4. Scale to 0-100% range.
        """
        if not self.is_trained:
            self.train()

        features_reshaped = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_reshaped)

        # predict_proba returns [P(not_compatible), P(compatible)]
        probability = self.model.predict_proba(features_scaled)[0][1]

        # Convert to percentage (0-100)
        return round(probability * 100, 2)

    def predict_batch(self, features_batch):
        """
        Predict compatibility for multiple user pairs at once.
        
        Args:
            features_batch (numpy.ndarray): Feature matrix of shape (n_pairs, 4)
        
        Returns:
            list[float]: List of compatibility percentages
        """
        if not self.is_trained:
            self.train()

        features_scaled = self.scaler.transform(features_batch)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        return [round(p * 100, 2) for p in probabilities]


# ============================================================================
# MODULE 5: Team Balance Analyzer
# ============================================================================

class TeamBalanceAnalyzer:
    """
    Analyzes team composition and identifies skill gaps.
    
    A balanced hackathon team typically needs:
        - AI/ML: Data processing, model building, predictions
        - Frontend: User interface, user experience
        - Backend: Server logic, APIs, infrastructure
        - Database: Data storage, queries, optimization
        - UI/UX: Design, wireframing, user research
    
    This module:
        1. Maps each user's skills to role categories.
        2. Identifies which roles are covered and which are missing.
        3. Calculates a team strength score.
        4. Suggests specific improvements.
    """

    # Define skill-to-role mapping
    ROLE_SKILLS = {
        'AI/ML': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'numpy', 'pandas',
                   'machine learning', 'deep learning', 'nlp', 'computer vision', 'keras',
                   'opencv', 'data science', 'ai', 'ml', 'statistics'],
        'Frontend': ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'typescript',
                      'tailwind', 'bootstrap', 'next.js', 'svelte', 'jquery', 'sass',
                      'webpack', 'vite'],
        'Backend': ['node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
                     'ruby', 'php', 'go', 'rust', 'java', 'c#', '.net', 'graphql',
                     'docker', 'kubernetes', 'microservices'],
        'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'redis',
                      'sqlite', 'dynamodb', 'cassandra', 'elasticsearch'],
        'UI/UX': ['figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator',
                   'ui design', 'ux design', 'wireframing', 'prototyping']
    }

    @staticmethod
    def analyze_team(users):
        """
        Analyze the skill coverage of a team.
        
        Args:
            users (list[dict]): List of user profiles with 'skills' field
        
        Returns:
            dict: Analysis result containing:
                - covered_roles: Roles that have at least one team member
                - missing_roles: Roles with no team members
                - role_coverage: Detailed breakdown of who covers each role
                - team_strength: Overall team strength level
                - recommendations: Specific improvement suggestions
        """
        # Collect all team skills (flattened, lowercase)
        all_skills = []
        for user in users:
            skills = [s.strip().lower() for s in user.get('skills', '').split(',')]
            all_skills.extend(skills)
        all_skills = set(all_skills)

        # Check which roles are covered
        role_coverage = {}
        covered_roles = []
        missing_roles = []

        for role, role_skills in TeamBalanceAnalyzer.ROLE_SKILLS.items():
            matching_skills = [s for s in all_skills if s in role_skills]
            coverage_pct = len(matching_skills) / max(len(role_skills), 1) * 100

            role_coverage[role] = {
                'covered': len(matching_skills) > 0,
                'matching_skills': matching_skills,
                'coverage_percentage': round(coverage_pct, 1),
                'skill_count': len(matching_skills)
            }

            if len(matching_skills) > 0:
                covered_roles.append(role)
            else:
                missing_roles.append(role)

        # Calculate team strength
        coverage_ratio = len(covered_roles) / len(TeamBalanceAnalyzer.ROLE_SKILLS)
        if coverage_ratio >= 0.8:
            strength = 'Strong'
        elif coverage_ratio >= 0.6:
            strength = 'Moderate'
        elif coverage_ratio >= 0.4:
            strength = 'Developing'
        else:
            strength = 'Weak'

        # Generate recommendations
        recommendations = []
        for role in missing_roles:
            sample_skills = TeamBalanceAnalyzer.ROLE_SKILLS[role][:3]
            recommendations.append(
                f"Add a team member with {role} skills (e.g., {', '.join(sample_skills)})"
            )

        if len(covered_roles) == len(TeamBalanceAnalyzer.ROLE_SKILLS):
            recommendations.append("Excellent! Your team covers all essential roles.")

        return {
            'covered_roles': covered_roles,
            'missing_roles': missing_roles,
            'role_coverage': role_coverage,
            'team_strength_level': strength,
            'coverage_score': round(coverage_ratio * 100, 1),
            'recommendations': recommendations,
            'total_unique_skills': len(all_skills)
        }


# ============================================================================
# MAIN AI PIPELINE — Orchestrates all modules
# ============================================================================

class SkillForgeAI:
    """
    Main AI pipeline that orchestrates all modules for the SkillForge system.
    
    Pipeline Flow:
        1. Load all user data from database.
        2. Vectorize skills using TF-IDF (Module 1).
        3. Find similar users using Cosine Similarity (Module 2).
        4. Engineer features for each candidate pair (Module 3).
        5. Predict compatibility using Logistic Regression (Module 4).
        6. Analyze team balance (Module 5).
        7. Return ranked recommendations with scores.
    """

    def __init__(self):
        self.vectorizer = SkillVectorizer()
        self.similarity_engine = SimilarityEngine()
        self.feature_engineer = FeatureEngineer()
        self.compatibility_predictor = CompatibilityPredictor()
        self.balance_analyzer = TeamBalanceAnalyzer()

        # Train the compatibility model on initialization
        self.compatibility_predictor.train()
        print("[AI PIPELINE] SkillForge AI initialized and ready.")

    def analyze_and_recommend(self, target_user, all_users, top_n=5):
        """
        Full AI pipeline: Find similar users and predict compatibility.
        
        Args:
            target_user (dict): The user seeking teammates
            all_users (list[dict]): All registered users
            top_n (int): Number of recommendations to return
        
        Returns:
            dict: Complete recommendation response with scores
        """
        if len(all_users) < 2:
            return {
                'recommended_teammates': [],
                'missing_skills': [],
                'team_strength_level': 'Insufficient Data',
                'message': 'Need at least 2 users for recommendations.'
            }

        # STEP 1: Prepare skill texts for TF-IDF vectorization
        skill_texts = [user['skills'] for user in all_users]
        target_skills = target_user['skills']

        # STEP 2: TF-IDF Vectorization (Module 1)
        # Fit vectorizer on ALL user skills (including target)
        all_skill_texts = skill_texts + [target_skills]
        all_vectors = self.vectorizer.fit_transform(all_skill_texts)

        # Separate target vector from existing users' vectors
        target_vector = all_vectors[-1]  # Last row is the target user
        user_vectors = all_vectors[:-1]  # All other rows

        # STEP 3: Find Top Similar Users (Module 2)
        user_ids = [user['id'] for user in all_users]
        top_similar = self.similarity_engine.find_top_similar(
            target_vector, user_vectors, user_ids,
            top_n=top_n,
            exclude_id=target_user.get('id')
        )

        # STEP 4: Feature Engineering + Compatibility Prediction (Modules 3 & 4)
        recommended_teammates = []
        recommended_users = []

        for rec_user_id, sim_score in top_similar:
            # Find the user data
            rec_user = next((u for u in all_users if u['id'] == rec_user_id), None)
            if rec_user is None:
                continue

            # Compute engineered features (Module 3)
            features = self.feature_engineer.compute_features(
                target_user, rec_user, sim_score
            )

            # Predict compatibility (Module 4)
            compatibility = self.compatibility_predictor.predict_compatibility(features)

            recommended_teammates.append({
                'id': rec_user['id'],
                'name': rec_user['name'],
                'skills': rec_user['skills'],
                'experience': rec_user['experience'],
                'domain': rec_user['domain'],
                'interest': rec_user['interest'],
                'similarity_score': round(sim_score * 100, 2),
                'compatibility_score': compatibility,
                'feature_details': {
                    'skill_similarity': round(sim_score, 4),
                    'experience_difference': int(features[1]),
                    'domain_match': bool(features[2]),
                    'interest_match': bool(features[3])
                }
            })
            recommended_users.append(rec_user)

        # Sort by compatibility score (highest first)
        recommended_teammates.sort(key=lambda x: x['compatibility_score'], reverse=True)

        # STEP 5: Team Balance Analysis (Module 5)
        team_members = [target_user] + recommended_users
        balance = self.balance_analyzer.analyze_team(team_members)

        return {
            'recommended_teammates': recommended_teammates,
            'missing_skills': balance['missing_roles'],
            'team_strength_level': balance['team_strength_level'],
            'coverage_score': balance['coverage_score'],
            'role_coverage': balance['role_coverage'],
            'recommendations': balance['recommendations'],
            'ai_metadata': {
                'tfidf_features': len(self.vectorizer.get_feature_names()),
                'model_type': 'Logistic Regression',
                'total_users_analyzed': len(all_users)
            }
        }

    def get_team_balance(self, users):
        """
        Analyze team balance for a group of users.
        
        Args:
            users (list[dict]): Team members
        
        Returns:
            dict: Team balance analysis
        """
        return self.balance_analyzer.analyze_team(users)
