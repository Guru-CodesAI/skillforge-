import { useState, useEffect, useCallback } from 'react';
import RegisterForm from './components/RegisterForm';
import RecommendationCard from './components/RecommendationCard';
import TeamBalancePanel from './components/TeamBalancePanel';
import Dashboard from './components/Dashboard';
import {
  analyzeUser,
  getTeamBalance,
  healthCheck,
  type AnalysisResult,
  type TeamBalance,
} from './api';

type Page = 'home' | 'analyze' | 'dashboard' | 'team-balance';

interface Toast {
  message: string;
  type: 'success' | 'error';
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [teamBalance, setTeamBalance] = useState<TeamBalance | null>(null);
  const [loading, setLoading] = useState(false);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [toast, setToast] = useState<Toast | null>(null);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setApiStatus('online');
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
  }, []);

  // Auto-dismiss toast
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
  };

  const handleAnalyze = useCallback(async (data: {
    name: string;
    skills: string;
    experience: string;
    domain: string;
    interest: string;
  }) => {
    setLoading(true);
    try {
      const result = await analyzeUser(data);
      setAnalysisResult(result);
      showToast(`Found ${result.recommended_teammates.length} teammates for ${data.name}!`, 'success');

      // Auto-load team balance
      if (result.analyzed_user) {
        setBalanceLoading(true);
        const recommendedIds = result.recommended_teammates.map(t => t.id);
        const allIds = [result.analyzed_user.id, ...recommendedIds];
        try {
          const balance = await getTeamBalance(allIds);
          setTeamBalance(balance);
        } catch {
          console.error('Failed to load team balance');
        }
        setBalanceLoading(false);
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed. Is the backend running?';
      showToast(errorMessage, 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div>
      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast-${toast.type}`} id="toast-notification">
          {toast.type === 'success' ? '✅' : '❌'} {toast.message}
        </div>
      )}

      {/* Navigation */}
      <nav className="nav" id="main-nav">
        <div className="nav-brand">
          <div className="nav-logo">⚡</div>
          <div>
            <div className="nav-title">SkillForge</div>
            <div className="nav-subtitle">AI Hackathon Teammate</div>
          </div>
        </div>
        <div className="nav-links">
          <button
            className={`nav-link ${currentPage === 'home' ? 'active' : ''}`}
            onClick={() => setCurrentPage('home')}
            id="nav-home"
          >
            🏠 Home
          </button>
          <button
            className={`nav-link ${currentPage === 'analyze' ? 'active' : ''}`}
            onClick={() => setCurrentPage('analyze')}
            id="nav-analyze"
          >
            🔍 Find Teammates
          </button>
          <button
            className={`nav-link ${currentPage === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentPage('dashboard')}
            id="nav-dashboard"
          >
            📊 Dashboard
          </button>
          <button
            className={`nav-link ${currentPage === 'team-balance' ? 'active' : ''}`}
            onClick={() => setCurrentPage('team-balance')}
            id="nav-team-balance"
          >
            ⚖️ Team Balance
          </button>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            borderRadius: 'var(--radius-full)',
            fontSize: '0.75rem',
            fontWeight: 600,
            background: apiStatus === 'online' ? 'rgba(6,214,160,0.1)' : apiStatus === 'offline' ? 'rgba(247,37,133,0.1)' : 'rgba(255,158,0,0.1)',
            color: apiStatus === 'online' ? 'var(--accent-cyan)' : apiStatus === 'offline' ? 'var(--accent-pink)' : 'var(--accent-orange)',
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: apiStatus === 'online' ? 'var(--accent-cyan)' : apiStatus === 'offline' ? 'var(--accent-pink)' : 'var(--accent-orange)',
              animation: apiStatus === 'checking' ? 'pulse 1s infinite' : 'none',
            }}></span>
            {apiStatus === 'online' ? 'API Online' : apiStatus === 'offline' ? 'API Offline' : 'Checking...'}
          </div>
        </div>
      </nav>

      <div className="app-container">
        {/* ==================== HOME PAGE ==================== */}
        {currentPage === 'home' && (
          <div className="page-content" id="page-home">
            <div className="hero">
              <h1 className="hero-title">
                Find Your Perfect<br />
                <span className="gradient">Hackathon Teammate</span>
              </h1>
              <p className="hero-description">
                SkillForge uses advanced AI — TF-IDF Vectorization, Cosine Similarity,
                and Logistic Regression — to intelligently match you with compatible
                teammates based on skills, experience, and interests.
              </p>
              <div className="hero-cta">
                <button className="btn btn-primary btn-lg" onClick={() => setCurrentPage('analyze')}>
                  🚀 Get Started
                </button>
                <button className="btn btn-secondary btn-lg" onClick={() => setCurrentPage('dashboard')}>
                  📊 View Dashboard
                </button>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-3" style={{ marginTop: 'var(--space-2xl)' }}>
              <div className="card" style={{ animation: 'fadeInUp 0.5s ease-out 0.1s backwards' }}>
                <div className="card-header">
                  <div className="card-icon primary">🧠</div>
                  <div>
                    <div className="card-title">TF-IDF Vectorization</div>
                  </div>
                </div>
                <p style={{ fontSize: '0.9rem' }}>
                  Converts your skill text into numerical vectors using Term Frequency-Inverse Document Frequency.
                  Rare skills get higher weight, making matching more meaningful.
                </p>
              </div>

              <div className="card" style={{ animation: 'fadeInUp 0.5s ease-out 0.2s backwards' }}>
                <div className="card-header">
                  <div className="card-icon accent">🎯</div>
                  <div>
                    <div className="card-title">Cosine Similarity</div>
                  </div>
                </div>
                <p style={{ fontSize: '0.9rem' }}>
                  Measures the angular similarity between skill vectors to find users with the most similar
                  skill profiles, regardless of how many skills they have.
                </p>
              </div>

              <div className="card" style={{ animation: 'fadeInUp 0.5s ease-out 0.3s backwards' }}>
                <div className="card-header">
                  <div className="card-icon purple">📊</div>
                  <div>
                    <div className="card-title">ML Prediction</div>
                  </div>
                </div>
                <p style={{ fontSize: '0.9rem' }}>
                  A trained Logistic Regression model predicts compatibility probability using engineered features like skill similarity, experience gap, and domain match.
                </p>
              </div>
            </div>

          </div>
        )}

        {/* ==================== ANALYZE PAGE ==================== */}
        {currentPage === 'analyze' && (
          <div className="page-content" id="page-analyze">
            <div className="section-header" style={{ textAlign: 'center', marginBottom: 'var(--space-2xl)' }}>
              <h2 className="section-title" style={{ justifyContent: 'center' }}>
                🔍 Find Your Teammates
              </h2>
              <p className="section-description" style={{ margin: '0 auto' }}>
                Enter your profile and let the AI find your most compatible hackathon teammates
              </p>
            </div>

            {apiStatus === 'offline' && (
              <div className="card" style={{
                maxWidth: 680,
                margin: '0 auto var(--space-xl)',
                borderColor: 'rgba(247,37,133,0.3)',
                background: 'rgba(247,37,133,0.05)',
                textAlign: 'center',
              }}>
                <p style={{ color: 'var(--accent-pink)', fontWeight: 600 }}>
                  ⚠️ Backend API is offline. Please start the Flask server first:
                </p>
                <code style={{
                  display: 'block',
                  marginTop: 'var(--space-md)',
                  padding: 'var(--space-md)',
                  background: 'var(--bg-secondary)',
                  borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem',
                  color: 'var(--accent-cyan)',
                }}>
                  cd backend && pip install -r requirements.txt && python app.py
                </code>
              </div>
            )}

            <RegisterForm onAnalyze={handleAnalyze} loading={loading} />

            {/* Analysis Results */}
            {analysisResult && analysisResult.recommended_teammates.length > 0 && (
              <div style={{ marginTop: 'var(--space-2xl)' }}>
                <div className="section-header" style={{ textAlign: 'center' }}>
                  <h3 className="section-title" style={{ justifyContent: 'center' }}>
                    🎯 AI-Recommended Teammates
                  </h3>
                  <p className="section-description" style={{ margin: '0 auto' }}>
                    Based on {analysisResult.ai_metadata?.tfidf_features} TF-IDF features analyzed across {analysisResult.ai_metadata?.total_users_analyzed} users
                  </p>
                </div>

                {/* Quick Summary Cards */}
                <div className="grid grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
                  <div className="stat-card accent" style={{ textAlign: 'center' }}>
                    <div className="stat-value" style={{ color: 'var(--accent-cyan)' }}>
                      {analysisResult.recommended_teammates.length}
                    </div>
                    <div className="stat-label">Matches Found</div>
                  </div>
                  <div className="stat-card primary" style={{ textAlign: 'center' }}>
                    <div className="stat-value gradient-text">
                      {analysisResult.recommended_teammates[0]?.compatibility_score.toFixed(0)}%
                    </div>
                    <div className="stat-label">Best Compatibility</div>
                  </div>
                  <div className="stat-card purple" style={{ textAlign: 'center' }}>
                    <div className="stat-value" style={{ color: 'var(--accent-purple)' }}>
                      {analysisResult.ai_metadata?.model_type}
                    </div>
                    <div className="stat-label" style={{ fontSize: '0.7rem' }}>ML Model Used</div>
                  </div>
                </div>

                {/* Recommendation Cards */}
                <div className="grid" style={{ gap: 'var(--space-lg)' }}>
                  {analysisResult.recommended_teammates.map((teammate, i) => (
                    <RecommendationCard key={teammate.id} teammate={teammate} rank={i + 1} />
                  ))}
                </div>

                {/* Missing Skills / Team Gaps */}
                {analysisResult.missing_skills && analysisResult.missing_skills.length > 0 && (
                  <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                    <div className="card-header">
                      <div className="card-icon pink">⚠️</div>
                      <div>
                        <div className="card-title">Missing Skill Roles</div>
                        <div className="card-subtitle">Roles not covered by the recommended team</div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-sm">
                      {analysisResult.missing_skills.map((skill, i) => (
                        <span key={i} className="badge badge-pink">{skill}</span>
                      ))}
                    </div>
                    <div style={{ marginTop: 'var(--space-md)' }}>
                      <span className={`strength-badge ${analysisResult.team_strength_level.toLowerCase()}`}>
                        Team Strength: {analysisResult.team_strength_level}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ==================== DASHBOARD PAGE ==================== */}
        {currentPage === 'dashboard' && (
          <div className="page-content" id="page-dashboard">
            <div className="section-header">
              <h2 className="section-title">📊 Dashboard</h2>
              <p className="section-description">
                Overview of all users, skills, and system analytics
              </p>
            </div>
            <Dashboard />
          </div>
        )}

        {/* ==================== TEAM BALANCE PAGE ==================== */}
        {currentPage === 'team-balance' && (
          <div className="page-content" id="page-team-balance">
            <div className="section-header">
              <h2 className="section-title">⚖️ Team Balance Analysis</h2>
              <p className="section-description">
                Analyze skill coverage and identify missing roles in your team
              </p>
            </div>
            <TeamBalancePanel balance={teamBalance} loading={balanceLoading} />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: 'var(--space-xl) var(--space-lg)',
        borderTop: '1px solid var(--border-subtle)',
        marginTop: 'var(--space-3xl)',
      }}>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          SkillForge – AI Hackathon Teammate • B.Tech AI & Data Science Project
        </p>
        <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
          Powered by TF-IDF • Cosine Similarity • Logistic Regression • scikit-learn
        </p>
      </footer>
    </div>
  );
}
