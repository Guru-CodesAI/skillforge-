# ⚡ SkillForge – AI Hackathon Teammate

> **AI-Powered Teammate Recommendation & Compatibility Prediction System**  
> B.Tech AI & Data Science – 2nd Year Academic Project

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![React](https://img.shields.io/badge/React-19-blue?logo=react)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [AI Workflow](#-ai-workflow)
3. [Feature Engineering](#-feature-engineering)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Setup & Run](#-setup--run)
7. [API Endpoints](#-api-endpoints)
8. [Database Schema](#-database-schema)
9. [Viva Questions & Answers](#-viva-questions--answers)
10. [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

**SkillForge** is an intelligent system that helps hackathon organizers and participants find the most compatible teammates. It uses:

- **NLP (TF-IDF Vectorization)** to convert skill descriptions into numerical vectors
- **Cosine Similarity** to find users with similar skill profiles
- **Logistic Regression** to predict team compatibility
- **Feature Engineering** to create structured inputs for the ML model
- **Team Balance Analysis** to identify skill gaps

**All AI is implemented locally using scikit-learn — no external AI APIs are used.**

---

## 🧠 AI Workflow

### Complete Pipeline

```
User Input (name, skills, experience, domain, interest)
     │
     ▼
┌─────────────────────────────────┐
│ MODULE 1: Skill Vectorization   │
│ TfidfVectorizer from sklearn    │
│ Input: "python, react, flask"   │
│ Output: Sparse TF-IDF vector    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ MODULE 2: Similarity Engine     │
│ cosine_similarity from sklearn  │
│ Compare target vs all users     │
│ Output: Top 5 similar users     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ MODULE 3: Feature Engineering   │
│ Create structured features:     │
│ • skill_similarity (float)      │
│ • experience_difference (int)   │
│ • domain_match (binary)         │
│ • interest_match (binary)       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ MODULE 4: Compatibility Model   │
│ Logistic Regression from sklearn│
│ Input: Feature vector [4 dims]  │
│ Output: Probability (0-100%)    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ MODULE 5: Team Balance Analyzer │
│ Maps skills → roles             │
│ Identifies covered/missing roles│
│ Calculates team strength score  │
│ Suggests improvements           │
└─────────────────────────────────┘
```

### AI Workflow Explanation

1. **TF-IDF Vectorization**: Converts skill text into weighted numerical vectors. Common skills (like "python") get lower weights, while specialized skills (like "computer vision") get higher weights. This ensures that unique expertise is emphasized in matching.

2. **Cosine Similarity**: Measures the angle between two TF-IDF vectors. A value of 1.0 means identical skill profiles; 0.0 means completely different. It's independent of vector magnitude, so users with different numbers of skills can still be compared fairly.

3. **Feature Engineering**: Transforms raw user data into ML-ready features. We create 4 features: skill similarity score, experience level difference, domain match flag, and interest match flag.

4. **Logistic Regression**: A supervised classification model that takes the 4 engineered features and predicts the probability of compatibility. It uses the sigmoid function to map linear combinations to probabilities.

5. **Team Balance Analysis**: Categorizes skills into 5 roles (AI/ML, Frontend, Backend, Database, UI/UX) and identifies which roles are covered and which are missing.

---

## ⚙️ Feature Engineering

### Engineered Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `skill_similarity` | Float | 0.0 - 1.0 | Cosine similarity between TF-IDF vectors |
| `experience_difference` | Integer | 0 - 2 | Absolute difference in experience levels |
| `domain_match` | Binary | 0 or 1 | Whether users share the same domain |
| `interest_match` | Binary | 0 or 1 | Whether users share the same hackathon interest |

### Why These Features?

- **Skill Similarity (weight: 0.35)**: The most important factor. Users with complementary skills work better in teams.
- **Experience Difference (weight: 0.20)**: Teams with mixed experience levels (mentor + mentee) often perform well.
- **Domain Match (weight: 0.25)**: People in the same domain understand each other's tools and terminology.
- **Interest Match (weight: 0.20)**: Shared hackathon interests ensure higher motivation and alignment.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 19 + TypeScript | UI Components |
| Build Tool | Vite 7 | Fast development server |
| Styling | Vanilla CSS | Dark theme, glassmorphism |
| HTTP Client | Axios | API communication |
| Backend | Flask 3.0 | REST API |
| ML Library | scikit-learn 1.3 | TF-IDF, Cosine Sim, Logistic Regression |
| Data | NumPy, Pandas | Numerical operations |
| Database | SQLite | Persistent storage |
| CORS | Flask-CORS | Cross-origin requests |

---

## 📁 Project Structure

```
SkillForge/
├── backend/
│   ├── app.py              # Flask REST API (Application Layer)
│   ├── model.py            # AI Modules (5 ML components)
│   ├── database.py         # SQLite operations (Data Layer)
│   ├── requirements.txt    # Python dependencies
│   └── skillforge.db       # SQLite database (auto-created)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RegisterForm.tsx        # User registration form
│   │   │   ├── RecommendationCard.tsx  # Teammate card with scores
│   │   │   ├── TeamBalancePanel.tsx    # Team analysis view
│   │   │   └── Dashboard.tsx           # Analytics dashboard
│   │   ├── api.ts           # API client with TypeScript types
│   │   ├── App.tsx          # Main application component
│   │   ├── App.css          # Component styles
│   │   ├── main.tsx         # React entry point
│   │   └── index.css        # Global design system
│   ├── index.html           # HTML template
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.ts       # Vite configuration
│   └── tsconfig.json        # TypeScript configuration
│
└── README.md                # This file
```

---

## 🚀 Setup & Run

### Prerequisites

- **Python 3.9+** installed
- **Node.js 18+** and **npm** installed
- **VS Code** (recommended)

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the Flask API server
python app.py
```

The backend will start on `http://localhost:5000`.  
You should see the AI model training logs in the terminal.

### Step 2: Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install Node.js dependencies
npm install

# Start the Vite dev server
npm run dev
```

The frontend will start on `http://localhost:5173`.

### 🔧 Changing Ports

If the default ports are in use, you can specify different ones:

**Frontend (React):**
```bash
cd frontend
npm run dev -- --port 3000
```

**Backend (Flask):**
If you run the backend on a different port (e.g., 5002), update the frontend configuration by creating a `.env` file in the `frontend` folder:
```env
VITE_API_URL=http://localhost:5002/api
```

### Step 3: Open in Browser

Visit `http://localhost:5173` (or your custom port) in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/register` | Register a new user |
| `POST` | `/api/analyze` | Run AI analysis for a user |
| `GET` | `/api/recommendations?user_id=1` | Get stored recommendations |
| `GET` | `/api/team-balance?user_ids=1,2,3` | Analyze team balance |
| `GET` | `/api/users` | List all users |
| `GET` | `/api/stats` | Get dashboard statistics |

### Example: Register & Analyze

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "skills": "python, react, machine learning",
    "experience": "intermediate",
    "domain": "ai_ml",
    "interest": "healthcare"
  }'
```

### Response Format

```json
{
  "success": true,
  "recommended_teammates": [
    {
      "name": "Arjun Sharma",
      "similarity_score": 72.5,
      "compatibility_score": 85.3,
      "feature_details": {
        "skill_similarity": 0.725,
        "experience_difference": 1,
        "domain_match": true,
        "interest_match": true
      }
    }
  ],
  "missing_skills": ["UI/UX"],
  "team_strength_level": "Strong"
}
```

---

## 🗃️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    skills TEXT NOT NULL,          -- Comma-separated skill string
    experience TEXT DEFAULT 'beginner',  -- beginner/intermediate/advanced
    domain TEXT DEFAULT 'general',       -- ai_ml/web_dev/mobile/etc
    interest TEXT DEFAULT 'general',     -- healthcare/fintech/edtech/etc
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Skills Table
```sql
CREATE TABLE skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    skill_name TEXT NOT NULL,
    category TEXT DEFAULT 'general',  -- ai_ml/frontend/backend/database/ui_ux
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Recommendations Table
```sql
CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    recommended_user_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    compatibility_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (recommended_user_id) REFERENCES users(id)
);
```

---

## 🎓 Viva Questions & Answers

### Q1: What is TF-IDF and why did you use it?
**A:** TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that measures how important a word is to a document in a collection. We use it to convert skill text into vectors. TF measures how often a skill appears, while IDF penalizes skills that are too common across all users. This gives higher weight to specialized skills like "computer vision" vs common ones like "python".

### Q2: How does Cosine Similarity work?
**A:** Cosine similarity measures the cosine of the angle between two vectors. Formula: cos(θ) = (A·B) / (||A|| × ||B||). It ranges from 0 (completely different) to 1 (identical). We chose it over Euclidean distance because it's independent of vector magnitude — a user with 3 skills can be fairly compared to one with 10 skills.

### Q3: Why Logistic Regression for compatibility prediction?
**A:** Logistic Regression is ideal because: (1) It outputs probabilities (0-100%), perfect for compatibility scoring. (2) It's interpretable — we can see which features matter most via coefficients. (3) It works well with small, well-engineered feature sets. (4) It uses the sigmoid function: P = 1/(1+e^(-z)), mapping linear outputs to probabilities.

### Q4: What is Feature Engineering?
**A:** Feature Engineering is the process of creating new input variables from raw data that better represent the underlying patterns. We created 4 features: skill_similarity (cosine score), experience_difference (ordinal encoding), domain_match (binary), and interest_match (binary). These structured features help the ML model make better predictions than raw data alone.

### Q5: Explain your system architecture.
**A:** We use a Three-Tier Architecture: (1) Presentation Layer — React frontend with dark theme UI. (2) Application Layer — Flask backend with 5 AI modules. (3) Data Layer — SQLite database with 3 tables. The frontend communicates with the backend via REST API, and the backend handles all AI processing locally using scikit-learn.

### Q6: How do you handle the cold-start problem?
**A:** We seed the database with 15 sample users with diverse skills, domains, and experience levels. The Logistic Regression model is trained on 500 synthetic samples generated using domain knowledge about hackathon team dynamics. As real users register, the system improves naturally.

### Q7: What is StandardScaler and why is it needed?
**A:** StandardScaler transforms features to have zero mean and unit variance (z-score normalization). This is essential because our features have different scales: skill_similarity ranges 0-1, while experience_difference ranges 0-2. Without scaling, features with larger values would dominate the model's learning.

### Q8: What evaluation metric would you use?
**A:** For our binary compatibility classifier, we would use: Accuracy (overall correctness), Precision (of those predicted compatible, how many actually are), Recall (of actually compatible pairs, how many did we find), and F1-Score (harmonic mean of precision and recall). The training accuracy logged on startup gives initial confidence.

### Q9: How does the Team Balance Analyzer work?
**A:** It maps each team member's skills to 5 predefined roles (AI/ML, Frontend, Backend, Database, UI/UX) using keyword matching. It then calculates: covered roles (at least one member), missing roles (no members), coverage percentage per role, overall team strength (Strong/Moderate/Developing/Weak), and generates specific improvement suggestions.

### Q10: What makes this different from a simple keyword match?
**A:** Keyword matching just checks exact matches. Our system uses TF-IDF which assigns importance weights, cosine similarity which measures semantic closeness of skill profiles, and an ML model that considers multiple factors simultaneously (skills, experience, domain, interests). This produces much more meaningful recommendations.

---

## 🔮 Future Enhancements

1. **Deep Learning**: Replace Logistic Regression with a neural network for more complex compatibility patterns.

2. **Collaborative Filtering**: Use user interaction data (who worked well together) to improve recommendations.

3. **Real-time Chat**: Add WebSocket-based messaging for matched teammates.

4. **Skill Embedding**: Use Word2Vec or BERT embeddings instead of TF-IDF for richer skill representations.

5. **User Feedback Loop**: Allow users to rate recommendations, creating labeled training data for model improvement.

6. **Graph-based Recommendations**: Build a skill graph to understand relationships between skills (e.g., "Python" → "Flask" → "REST API").

7. **Resume Parsing**: Auto-extract skills from uploaded resumes using NLP.

8. **Deployment**: Deploy on AWS/Heroku with a production database (PostgreSQL).

9. **Authentication**: Add JWT-based user authentication and session management.

10. **Analytics Dashboard**: Advanced visualizations with D3.js showing skill networks and team dynamics.

---

## 📄 License

This project is developed for academic purposes as part of B.Tech AI & Data Science curriculum.

---

**Built with ❤️ using Python, React, and scikit-learn**
