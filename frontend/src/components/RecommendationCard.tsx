import { type RecommendedTeammate } from '../api';

interface RecommendationCardProps {
    teammate: RecommendedTeammate;
    rank: number;
}

export default function RecommendationCard({ teammate, rank }: RecommendationCardProps) {
    const initials = teammate.name
        .split(' ')
        .map(n => n[0])
        .join('')
        .toUpperCase();

    const compatClass = teammate.compatibility_score >= 70 ? 'high' : teammate.compatibility_score >= 45 ? 'medium' : 'low';

    const gradients = [
        'linear-gradient(135deg, #4361ee, #9b5de5)',
        'linear-gradient(135deg, #06d6a0, #00bbf9)',
        'linear-gradient(135deg, #f72585, #ff9e00)',
        'linear-gradient(135deg, #9b5de5, #f72585)',
        'linear-gradient(135deg, #00bbf9, #4361ee)',
    ];

    const domainLabels: Record<string, string> = {
        ai_ml: '🤖 AI/ML',
        web_dev: '🌐 Web Dev',
        mobile: '📱 Mobile',
        data_science: '📊 Data Science',
        ui_ux: '🎨 UI/UX',
        database: '🗄️ Database',
        devops: '⚙️ DevOps',
        general: '💻 General',
    };

    const experienceLabels: Record<string, string> = {
        beginner: '🌱 Beginner',
        intermediate: '🚀 Intermediate',
        advanced: '⚡ Advanced',
    };

    return (
        <div className="rec-card" id={`recommendation-${rank}`}>
            <div className="rec-header">
                <div className="rec-user">
                    <div
                        className="rec-avatar"
                        style={{ background: gradients[(rank - 1) % gradients.length] }}
                    >
                        {initials}
                    </div>
                    <div>
                        <div className="rec-name">
                            <span style={{ marginRight: 8 }}>#{rank}</span>
                            {teammate.name}
                        </div>
                        <div className="rec-domain">
                            {domainLabels[teammate.domain] || teammate.domain} • {experienceLabels[teammate.experience] || teammate.experience}
                        </div>
                    </div>
                </div>
                <div className="rec-scores">
                    <div className="rec-score">
                        <div className="rec-score-value similarity">
                            {teammate.similarity_score.toFixed(1)}%
                        </div>
                        <div className="rec-score-label">Similarity</div>
                    </div>
                    <div className="rec-score">
                        <div className="rec-score-value compatibility">
                            {teammate.compatibility_score.toFixed(1)}%
                        </div>
                        <div className="rec-score-label">Compatible</div>
                    </div>
                </div>
            </div>

            {/* Compatibility Progress Bar */}
            <div className="compatibility-bar-container">
                <div className="compatibility-bar-label">
                    <span>Compatibility Score</span>
                    <span style={{ color: compatClass === 'high' ? 'var(--accent-cyan)' : compatClass === 'medium' ? 'var(--primary-400)' : 'var(--accent-pink)' }}>
                        {teammate.compatibility_score.toFixed(1)}%
                    </span>
                </div>
                <div className="progress-bar">
                    <div
                        className={`progress-fill ${compatClass}`}
                        style={{ width: `${teammate.compatibility_score}%` }}
                    />
                </div>
            </div>

            {/* Skills */}
            <div className="rec-skills">
                {teammate.skills.split(',').map((skill, i) => (
                    <span key={i} className="skill-tag">{skill.trim()}</span>
                ))}
            </div>

            {/* Feature Details */}
            <div className="rec-features">
                <div className="rec-feature">
                    <span className={`rec-feature-icon ${teammate.feature_details.domain_match ? 'match' : 'no-match'}`}>
                        {teammate.feature_details.domain_match ? '✅' : '❌'}
                    </span>
                    Domain Match
                </div>
                <div className="rec-feature">
                    <span className={`rec-feature-icon ${teammate.feature_details.interest_match ? 'match' : 'no-match'}`}>
                        {teammate.feature_details.interest_match ? '✅' : '❌'}
                    </span>
                    Interest Match
                </div>
                <div className="rec-feature">
                    <span className="rec-feature-icon" style={{ color: 'var(--accent-blue)' }}>📊</span>
                    Skill Similarity: {(teammate.feature_details.skill_similarity * 100).toFixed(1)}%
                </div>
                <div className="rec-feature">
                    <span className="rec-feature-icon" style={{ color: 'var(--accent-orange)' }}>📈</span>
                    Exp. Diff: {teammate.feature_details.experience_difference}
                </div>
            </div>
        </div>
    );
}
