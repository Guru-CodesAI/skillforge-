"""
database.py — Data Layer for SkillForge
========================================
This module handles all SQLite database operations including:
- Database initialization and table creation
- User registration and retrieval
- Recommendation storage
- Skill-based queries

Architecture Note:
    This is the Data Layer in our Three-Tier Architecture.
    It abstracts all database operations behind clean function interfaces,
    ensuring the Application Layer (app.py) never writes raw SQL.
"""

import sqlite3
import os
import json
from datetime import datetime

# Database file path (stored in the backend directory)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skillforge.db')


def get_connection():
    """
    Create and return a database connection with Row factory enabled.
    Row factory allows accessing columns by name (dict-like) instead of index.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database schema.
    Creates all required tables if they don't exist.
    
    Tables:
        - users: Stores registered user profiles with skills and preferences
        - recommendations: Stores AI-generated teammate recommendations
        - skills: Stores normalized individual skills for analytics
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Users table — core profile data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            skills TEXT NOT NULL,
            experience TEXT NOT NULL DEFAULT 'beginner',
            domain TEXT NOT NULL DEFAULT 'general',
            interest TEXT NOT NULL DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Recommendations table — stores AI-generated recommendations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            recommended_user_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            compatibility_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (recommended_user_id) REFERENCES users(id)
        )
    ''')

    # Skills table — normalized skill entries for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            skill_name TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()


def add_user(name, skills, experience='beginner', domain='general', interest='general'):
    """
    Register a new user in the database.
    
    Args:
        name (str): User's full name
        skills (str): Comma-separated skill string (e.g., "python, react, tensorflow")
        experience (str): Experience level — 'beginner', 'intermediate', 'advanced'
        domain (str): Primary domain — 'ai_ml', 'web_dev', 'mobile', 'data_science', etc.
        interest (str): Hackathon interest — 'healthcare', 'fintech', 'edtech', etc.
    
    Returns:
        int: The ID of the newly created user
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO users (name, skills, experience, domain, interest)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, skills.lower().strip(), experience.lower().strip(), 
          domain.lower().strip(), interest.lower().strip()))

    user_id = cursor.lastrowid

    # Also store individual skills in the skills table for analytics
    skill_list = [s.strip() for s in skills.split(',') if s.strip()]

    # Categorize skills automatically
    skill_categories = {
        'ai_ml': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'numpy', 'pandas',
                   'machine learning', 'deep learning', 'nlp', 'computer vision', 'keras',
                   'opencv', 'data science', 'ai', 'ml'],
        'frontend': ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'typescript',
                      'tailwind', 'bootstrap', 'next.js', 'svelte', 'jquery', 'sass'],
        'backend': ['node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
                     'ruby', 'php', 'go', 'rust', 'java', 'c#', '.net', 'graphql'],
        'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'firebase', 'redis',
                      'sqlite', 'dynamodb', 'cassandra', 'elasticsearch'],
        'ui_ux': ['figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator',
                   'ui design', 'ux design', 'wireframing', 'prototyping']
    }

    for skill in skill_list:
        category = 'general'
        for cat, keywords in skill_categories.items():
            if skill.lower() in keywords:
                category = cat
                break
        cursor.execute('''
            INSERT INTO skills (user_id, skill_name, category)
            VALUES (?, ?, ?)
        ''', (user_id, skill.lower(), category))

    conn.commit()
    conn.close()
    return user_id


def get_all_users():
    """
    Retrieve all registered users.
    
    Returns:
        list[dict]: List of user dictionaries with all profile fields
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def get_user_by_id(user_id):
    """
    Retrieve a single user by their ID.
    
    Args:
        user_id (int): The user's database ID
    
    Returns:
        dict or None: User dictionary if found, None otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_recommendation(user_id, recommended_user_id, similarity_score, compatibility_score):
    """
    Store an AI-generated recommendation in the database.
    
    Args:
        user_id (int): The requesting user's ID
        recommended_user_id (int): The recommended teammate's ID
        similarity_score (float): Cosine similarity score (0-1)
        compatibility_score (float): ML-predicted compatibility (0-100)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recommendations (user_id, recommended_user_id, similarity_score, compatibility_score)
        VALUES (?, ?, ?, ?)
    ''', (user_id, recommended_user_id, similarity_score, compatibility_score))
    conn.commit()
    conn.close()


def get_recommendations_for_user(user_id):
    """
    Retrieve all recommendations for a specific user.
    
    Args:
        user_id (int): The user's database ID
    
    Returns:
        list[dict]: List of recommendation dictionaries with user details
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.*, u.name as recommended_name, u.skills as recommended_skills,
               u.experience as recommended_experience, u.domain as recommended_domain
        FROM recommendations r
        JOIN users u ON r.recommended_user_id = u.id
        WHERE r.user_id = ?
        ORDER BY r.compatibility_score DESC
    ''', (user_id,))
    recs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return recs


def get_user_skills_by_category(user_id):
    """
    Get a user's skills grouped by category.
    
    Args:
        user_id (int): The user's database ID
    
    Returns:
        dict: Skills grouped by category
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT skill_name, category FROM skills WHERE user_id = ?', (user_id,))
    skills = {}
    for row in cursor.fetchall():
        cat = row['category']
        if cat not in skills:
            skills[cat] = []
        skills[cat].append(row['skill_name'])
    conn.close()
    return skills


def get_all_skills_distribution():
    """
    Get the distribution of all skills across all users.
    Useful for analytics and team balance analysis.
    
    Returns:
        dict: Category-wise skill counts
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT category, COUNT(*) as count
        FROM skills
        GROUP BY category
        ORDER BY count DESC
    ''')
    distribution = {row['category']: row['count'] for row in cursor.fetchall()}
    conn.close()
    return distribution


def seed_sample_data():
    """
    Seed the database with sample users for demonstration and testing.
    This provides realistic data for the AI model to work with.
    """
    sample_users = [
        {
            'name': 'Arjun Sharma',
            'skills': 'python, tensorflow, machine learning, pandas, numpy, data science',
            'experience': 'advanced',
            'domain': 'ai_ml',
            'interest': 'healthcare'
        },
        {
            'name': 'Priya Patel',
            'skills': 'react, javascript, typescript, html, css, tailwind, next.js',
            'experience': 'intermediate',
            'domain': 'web_dev',
            'interest': 'edtech'
        },
        {
            'name': 'Rahul Verma',
            'skills': 'python, flask, django, postgresql, redis, docker',
            'experience': 'advanced',
            'domain': 'web_dev',
            'interest': 'fintech'
        },
        {
            'name': 'Sneha Gupta',
            'skills': 'figma, adobe xd, ui design, ux design, prototyping, css',
            'experience': 'intermediate',
            'domain': 'ui_ux',
            'interest': 'healthcare'
        },
        {
            'name': 'Vikram Singh',
            'skills': 'python, pytorch, deep learning, nlp, computer vision, opencv',
            'experience': 'advanced',
            'domain': 'ai_ml',
            'interest': 'healthcare'
        },
        {
            'name': 'Ananya Reddy',
            'skills': 'node.js, express, mongodb, firebase, graphql, typescript',
            'experience': 'intermediate',
            'domain': 'web_dev',
            'interest': 'fintech'
        },
        {
            'name': 'Karthik Nair',
            'skills': 'react, vue, angular, javascript, html, css, bootstrap',
            'experience': 'beginner',
            'domain': 'web_dev',
            'interest': 'edtech'
        },
        {
            'name': 'Meera Iyer',
            'skills': 'python, scikit-learn, pandas, numpy, sql, data science, machine learning',
            'experience': 'intermediate',
            'domain': 'data_science',
            'interest': 'healthcare'
        },
        {
            'name': 'Aditya Kumar',
            'skills': 'java, spring, mysql, docker, kubernetes, aws',
            'experience': 'advanced',
            'domain': 'web_dev',
            'interest': 'fintech'
        },
        {
            'name': 'Divya Menon',
            'skills': 'python, tensorflow, keras, nlp, deep learning, pandas',
            'experience': 'intermediate',
            'domain': 'ai_ml',
            'interest': 'edtech'
        },
        {
            'name': 'Rohan Joshi',
            'skills': 'react, node.js, mongodb, express, javascript, html, css',
            'experience': 'intermediate',
            'domain': 'web_dev',
            'interest': 'healthcare'
        },
        {
            'name': 'Ishita Banerjee',
            'skills': 'sql, postgresql, mongodb, redis, elasticsearch, python',
            'experience': 'advanced',
            'domain': 'database',
            'interest': 'fintech'
        },
        {
            'name': 'Siddharth Rao',
            'skills': 'flutter, dart, firebase, react native, mobile development',
            'experience': 'intermediate',
            'domain': 'mobile',
            'interest': 'edtech'
        },
        {
            'name': 'Neha Saxena',
            'skills': 'python, machine learning, scikit-learn, tensorflow, statistics',
            'experience': 'beginner',
            'domain': 'ai_ml',
            'interest': 'healthcare'
        },
        {
            'name': 'Amit Tiwari',
            'skills': 'go, rust, docker, kubernetes, microservices, linux',
            'experience': 'advanced',
            'domain': 'web_dev',
            'interest': 'fintech'
        }
    ]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        for user in sample_users:
            add_user(**user)
        print(f"[DATABASE] Seeded {len(sample_users)} sample users successfully.")
    else:
        print(f"[DATABASE] Database already contains {count} users. Skipping seed.")


# Initialize database when module is imported
init_db()
