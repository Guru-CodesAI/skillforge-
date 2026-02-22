"""
Vercel Serverless Function — Wraps the Flask backend for deployment.
All /api/* routes are handled by this single serverless function.
"""

import sys
import os

# Add backend directory to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from database import init_db, add_user, get_all_users, get_user_by_id, \
    save_recommendation, get_recommendations_for_user, seed_sample_data, \
    get_user_skills_by_category, get_all_skills_distribution
from model import SkillForgeAI
import traceback

# ============================================================================
# Flask App Configuration (Serverless)
# ============================================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize AI Pipeline
seed_sample_data()
ai_engine = SkillForgeAI()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'SkillForge AI',
        'version': '1.0.0',
        'deployment': 'vercel'
    })


@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        name = data.get('name', '').strip()
        skills = data.get('skills', '').strip()

        if not name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400

        experience = data.get('experience', 'beginner').strip()
        domain = data.get('domain', 'general').strip()
        interest = data.get('interest', 'general').strip()

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        user_id = data.get('user_id')
        if user_id:
            target_user = get_user_by_id(user_id)
            if not target_user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
        else:
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

            new_id = add_user(name, skills, experience, domain, interest)
            target_user = get_user_by_id(new_id)

        all_users = get_all_users()

        results = ai_engine.analyze_and_recommend(
            target_user=target_user,
            all_users=all_users,
            top_n=5
        )

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
    try:
        user_ids_str = request.args.get('user_ids', '')

        if not user_ids_str:
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
    try:
        users = get_all_users()
        skills_dist = get_all_skills_distribution()

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
