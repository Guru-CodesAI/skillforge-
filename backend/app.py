"""
app.py — Flask REST API for SkillForge
========================================
This is the Application Layer in our Three-Tier Architecture.
It exposes REST API endpoints for:
    - User registration (POST /api/register)
    - AI analysis & recommendations (POST /api/analyze)
    - Fetching recommendations (GET /api/recommendations)
    - Team balance analysis (GET /api/team-balance)
    - User listing (GET /api/users)
    - Dashboard stats (GET /api/stats)

Architecture:
    Frontend (React) → Flask API → AI Module (model.py) → Database (database.py)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from database import init_db, add_user, get_all_users, get_user_by_id, \
    save_recommendation, get_recommendations_for_user, seed_sample_data, \
    get_user_skills_by_category, get_all_skills_distribution
from model import SkillForgeAI
import traceback

# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__)

# Enable CORS for all routes (allows React frontend to communicate)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize AI Pipeline (trains model on startup)
print("\n" + "=" * 60)
print("  SkillForge AI — Initializing...")
print("=" * 60)

# Seed database with sample data for demonstration
seed_sample_data()

# Initialize AI engine (this trains the Logistic Regression model)
ai_engine = SkillForgeAI()

print("=" * 60)
print("  SkillForge AI — Ready!")
print("=" * 60 + "\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'service': 'SkillForge AI',
        'version': '1.0.0'
    })


@app.route('/api/register', methods=['POST'])
def register_user():
    """
    POST /api/register
    
    Register a new user with their skills and preferences.
    
    Request Body (JSON):
        {
            "name": "John Doe",
            "skills": "python, react, tensorflow",
            "experience": "intermediate",
            "domain": "ai_ml",
            "interest": "healthcare"
        }
    
    Response:
        {
            "success": true,
            "user_id": 16,
            "message": "User registered successfully"
        }
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        name = data.get('name', '').strip()
        skills = data.get('skills', '').strip()

        if not name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400

        # Optional fields with defaults
        experience = data.get('experience', 'beginner').strip()
        domain = data.get('domain', 'general').strip()
        interest = data.get('interest', 'general').strip()

        # Register user in database
        user_id = add_user(name, skills, experience, domain, interest)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'message': f'User "{name}" registered successfully!'
        }), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_user():
    """
    POST /api/analyze
    
    Run AI analysis for a user — find compatible teammates.
    
    Request Body (JSON):
        {
            "user_id": 1
        }
    
    OR register + analyze in one step:
        {
            "name": "John Doe",
            "skills": "python, react, tensorflow",
            "experience": "intermediate",
            "domain": "ai_ml",
            "interest": "healthcare"
        }
    
    Response:
        {
            "success": true,
            "recommended_teammates": [...],
            "missing_skills": [...],
            "team_strength_level": "Strong"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Option 1: Analyze existing user by ID
        user_id = data.get('user_id')
        if user_id:
            target_user = get_user_by_id(user_id)
            if not target_user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
        else:
            # Option 2: Register new user and analyze
            name = data.get('name', '').strip()
            skills = data.get('skills', '').strip()

            if not name or not skills:
                return jsonify({
                    'success': False,
                    'error': 'Either user_id or (name + skills) is required'
                }), 400

            experience = data.get('experience', 'beginner')
            domain = data.get('domain', 'general')
            interest = data.get('interest', 'general')

            # Register user first
            new_id = add_user(name, skills, experience, domain, interest)
            target_user = get_user_by_id(new_id)

        # Get all users for comparison
        all_users = get_all_users()

        # Run AI pipeline
        results = ai_engine.analyze_and_recommend(
            target_user=target_user,
            all_users=all_users,
            top_n=5
        )

        # Save recommendations to database
        for rec in results.get('recommended_teammates', []):
            save_recommendation(
                user_id=target_user['id'],
                recommended_user_id=rec['id'],
                similarity_score=rec['similarity_score'] / 100,
                compatibility_score=rec['compatibility_score']
            )

        results['success'] = True
        results['analyzed_user'] = {
            'id': target_user['id'],
            'name': target_user['name'],
            'skills': target_user['skills'],
            'experience': target_user['experience'],
            'domain': target_user['domain'],
            'interest': target_user['interest']
        }

        return jsonify(results), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """
    GET /api/recommendations?user_id=1
    
    Retrieve stored recommendations for a user.
    
    Query Parameters:
        user_id (int): The user's database ID
    
    Response:
        {
            "success": true,
            "recommendations": [...]
        }
    """
    try:
        user_id = request.args.get('user_id', type=int)

        if not user_id:
            return jsonify({'success': False, 'error': 'user_id parameter is required'}), 400

        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        recommendations = get_recommendations_for_user(user_id)

        return jsonify({
            'success': True,
            'user': dict(user),
            'recommendations': recommendations
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/team-balance', methods=['GET'])
def team_balance():
    """
    GET /api/team-balance?user_ids=1,2,3
    
    Analyze the skill balance of a team.
    
    Query Parameters:
        user_ids (str): Comma-separated user IDs
    
    Response:
        {
            "success": true,
            "covered_roles": [...],
            "missing_roles": [...],
            "team_strength_level": "Strong",
            "recommendations": [...]
        }
    """
    try:
        user_ids_str = request.args.get('user_ids', '')

        if not user_ids_str:
            # If no user_ids provided, analyze all users
            users = get_all_users()
        else:
            user_ids = [int(uid.strip()) for uid in user_ids_str.split(',') if uid.strip()]
            users = [get_user_by_id(uid) for uid in user_ids]
            users = [u for u in users if u is not None]

        if not users:
            return jsonify({'success': False, 'error': 'No valid users found'}), 404

        balance = ai_engine.get_team_balance(users)
        balance['success'] = True
        balance['team_members'] = [{'id': u['id'], 'name': u['name'], 'skills': u['skills']} for u in users]

        return jsonify(balance), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
def list_users():
    """
    GET /api/users
    
    List all registered users.
    
    Response:
        {
            "success": true,
            "users": [...],
            "count": 15
        }
    """
    try:
        users = get_all_users()
        return jsonify({
            'success': True,
            'users': users,
            'count': len(users)
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    GET /api/stats
    
    Get dashboard statistics.
    
    Response:
        {
            "success": true,
            "total_users": 15,
            "skills_distribution": {...},
            "experience_breakdown": {...}
        }
    """
    try:
        users = get_all_users()
        skills_dist = get_all_skills_distribution()

        # Calculate experience breakdown
        experience_counts = {}
        domain_counts = {}
        for user in users:
            exp = user.get('experience', 'beginner')
            dom = user.get('domain', 'general')
            experience_counts[exp] = experience_counts.get(exp, 0) + 1
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        return jsonify({
            'success': True,
            'total_users': len(users),
            'skills_distribution': skills_dist,
            'experience_breakdown': experience_counts,
            'domain_breakdown': domain_counts
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Run the Flask Development Server
# ============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting SkillForge API Server...")
    print("📍 API Base URL: http://localhost:5002/api")
    print("📍 Health Check: http://localhost:5002/api/health")
    print("📍 Register:     POST http://localhost:5002/api/register")
    print("📍 Analyze:      POST http://localhost:5002/api/analyze")
    print("📍 Recommend:    GET  http://localhost:5002/api/recommendations?user_id=1")
    print("📍 Team Balance: GET  http://localhost:5002/api/team-balance?user_ids=1,2,3")
    print("📍 Users:        GET  http://localhost:5002/api/users")
    print("📍 Stats:        GET  http://localhost:5002/api/stats\n")

    app.run(debug=True, host='0.0.0.0', port=5002)
