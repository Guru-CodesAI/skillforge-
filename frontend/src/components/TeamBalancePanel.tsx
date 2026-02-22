import { type TeamBalance } from '../api';

interface TeamBalancePanelProps {
    balance: TeamBalance | null;
    loading: boolean;
}

const ROLE_ICONS: Record<string, string> = {
    'AI/ML': '🤖',
    'Frontend': '🎨',
    'Backend': '⚙️',
    'Database': '🗄️',
    'UI/UX': '✨',
};

export default function TeamBalancePanel({ balance, loading }: TeamBalancePanelProps) {
    if (loading) {
        return (
            <div className="loading-overlay">
                <div className="spinner"></div>
                <div className="loading-text">Analyzing team balance...</div>
            </div>
        );
    }

    if (!balance) {
        return (
            <div className="empty-state">
                <div className="empty-icon">⚖️</div>
                <div className="empty-title">No Team Analysis Yet</div>
                <div className="empty-description">
                    Register and analyze your profile to see team balance insights.
                </div>
            </div>
        );
    }

    const strengthClass = balance.team_strength_level.toLowerCase();

    return (
        <div id="team-balance-panel" style={{ animation: 'fadeIn 0.5s ease-out' }}>
            {/* Team Strength Header */}
            <div className="card" style={{ marginBottom: 'var(--space-lg)', textAlign: 'center' }}>
                <div style={{ marginBottom: 'var(--space-md)' }}>
                    <span className={`strength-badge ${strengthClass}`}>
                        {strengthClass === 'strong' && '💪'}
                        {strengthClass === 'moderate' && '👍'}
                        {strengthClass === 'developing' && '📈'}
                        {strengthClass === 'weak' && '⚠️'}
                        {balance.team_strength_level} Team
                    </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 'var(--space-2xl)', flexWrap: 'wrap' }}>
                    <div>
                        <div className="stat-value gradient-text" style={{ fontSize: '2rem' }}>
                            {balance.coverage_score}%
                        </div>
                        <div className="stat-label">Coverage Score</div>
                    </div>
                    <div>
                        <div className="stat-value" style={{ fontSize: '2rem', color: 'var(--accent-cyan)' }}>
                            {balance.covered_roles.length}/{balance.covered_roles.length + balance.missing_roles.length}
                        </div>
                        <div className="stat-label">Roles Covered</div>
                    </div>
                    <div>
                        <div className="stat-value" style={{ fontSize: '2rem', color: 'var(--accent-purple)' }}>
                            {balance.total_unique_skills}
                        </div>
                        <div className="stat-label">Unique Skills</div>
                    </div>
                </div>
            </div>

            {/* Role Coverage Grid */}
            <div className="section-header">
                <h3 className="section-title">🎯 Role Coverage</h3>
                <p className="section-description">Analysis of skill roles covered by your team</p>
            </div>

            <div className="role-grid" style={{ marginBottom: 'var(--space-xl)' }}>
                {balance.role_coverage && Object.entries(balance.role_coverage).map(([role, coverage]) => (
                    <div
                        key={role}
                        className={`role-card ${coverage.covered ? 'covered' : 'missing'}`}
                    >
                        <div className="role-name">
                            <span>{ROLE_ICONS[role] || '📌'}</span>
                            {role}
                            <span className={`role-status ${coverage.covered ? 'covered-label' : 'missing-label'}`}>
                                {coverage.covered ? '✓' : '✗'}
                            </span>
                        </div>
                        <div className="progress-bar" style={{ marginBottom: 'var(--space-sm)' }}>
                            <div
                                className={`progress-fill ${coverage.coverage_percentage >= 20 ? 'high' : coverage.coverage_percentage >= 10 ? 'medium' : 'low'}`}
                                style={{ width: `${Math.min(coverage.coverage_percentage, 100)}%` }}
                            />
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            {coverage.skill_count} skill{coverage.skill_count !== 1 ? 's' : ''} • {coverage.coverage_percentage}% coverage
                        </div>
                        {coverage.matching_skills.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '8px' }}>
                                {coverage.matching_skills.map((skill, i) => (
                                    <span key={i} className="skill-tag" style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Recommendations */}
            {balance.recommendations && balance.recommendations.length > 0 && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon accent">💡</div>
                        <div>
                            <div className="card-title">AI Recommendations</div>
                            <div className="card-subtitle">Suggestions to strengthen your team</div>
                        </div>
                    </div>
                    <div className="flex flex-col gap-sm">
                        {balance.recommendations.map((rec, i) => (
                            <div
                                key={i}
                                className="feature-detail"
                                style={{ animation: `fadeIn 0.4s ease-out ${i * 0.1}s backwards` }}
                            >
                                <span style={{ fontSize: '1.2rem' }}>
                                    {rec.includes('Excellent') ? '🎉' : '📌'}
                                </span>
                                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{rec}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Team Members */}
            {balance.team_members && balance.team_members.length > 0 && (
                <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
                    <div className="card-header">
                        <div className="card-icon purple">👥</div>
                        <div>
                            <div className="card-title">Team Members ({balance.team_members.length})</div>
                            <div className="card-subtitle">Users included in this analysis</div>
                        </div>
                    </div>
                    <div className="flex flex-col gap-sm">
                        {balance.team_members.map((member) => (
                            <div key={member.id} className="feature-detail">
                                <span style={{ fontWeight: 700, minWidth: 120 }}>{member.name}</span>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    {member.skills.split(',').slice(0, 5).map((skill, i) => (
                                        <span key={i} className="skill-tag" style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
                                            {skill.trim()}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
