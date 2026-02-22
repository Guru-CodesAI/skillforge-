import { useEffect, useState } from 'react';
import { getStats, getUsers, type Stats, type User } from '../api';

export default function Dashboard() {
    const [stats, setStats] = useState<Stats | null>(null);
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statsData, usersData] = await Promise.all([
                    getStats(),
                    getUsers(),
                ]);
                setStats(statsData);
                setUsers(usersData.users);
            } catch {
                console.error('Failed to load dashboard data');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) {
        return (
            <div className="loading-overlay">
                <div className="spinner"></div>
                <div className="loading-text">Loading dashboard...</div>
            </div>
        );
    }

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

    const categoryColors: Record<string, string> = {
        ai_ml: '#9b5de5',
        frontend: '#4361ee',
        backend: '#06d6a0',
        database: '#ff9e00',
        ui_ux: '#f72585',
        general: '#6b7280',
    };

    return (
        <div id="dashboard" style={{ animation: 'fadeIn 0.5s ease-out' }}>
            {/* Stats Cards */}
            <div className="grid grid-4" style={{ marginBottom: 'var(--space-xl)' }}>
                <div className="stat-card primary">
                    <div className="stat-value gradient-text">{stats?.total_users || 0}</div>
                    <div className="stat-label">Total Users</div>
                </div>
                <div className="stat-card accent">
                    <div className="stat-value" style={{ color: 'var(--accent-cyan)' }}>
                        {Object.values(stats?.skills_distribution || {}).reduce((a, b) => a + b, 0)}
                    </div>
                    <div className="stat-label">Total Skills</div>
                </div>
                <div className="stat-card purple">
                    <div className="stat-value" style={{ color: 'var(--accent-purple)' }}>
                        {Object.keys(stats?.domain_breakdown || {}).length}
                    </div>
                    <div className="stat-label">Domains</div>
                </div>
                <div className="stat-card orange">
                    <div className="stat-value" style={{ color: 'var(--accent-orange)' }}>
                        {Object.keys(stats?.skills_distribution || {}).length}
                    </div>
                    <div className="stat-label">Skill Categories</div>
                </div>
            </div>

            <div className="grid grid-2">
                {/* Skills Distribution */}
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon primary">📊</div>
                        <div>
                            <div className="card-title">Skills Distribution</div>
                            <div className="card-subtitle">Skills by category across all users</div>
                        </div>
                    </div>
                    <div className="flex flex-col gap-md">
                        {stats?.skills_distribution && Object.entries(stats.skills_distribution)
                            .sort(([, a], [, b]) => b - a)
                            .map(([category, count]) => {
                                const maxCount = Math.max(...Object.values(stats.skills_distribution));
                                const pct = (count / maxCount) * 100;
                                return (
                                    <div key={category}>
                                        <div className="flex items-center justify-between" style={{ marginBottom: 6 }}>
                                            <span style={{ fontSize: '0.85rem', fontWeight: 600, textTransform: 'capitalize' }}>
                                                {category.replace('_', '/')}
                                            </span>
                                            <span style={{
                                                fontSize: '0.8rem',
                                                fontFamily: 'var(--font-mono)',
                                                color: categoryColors[category] || 'var(--text-secondary)'
                                            }}>
                                                {count}
                                            </span>
                                        </div>
                                        <div className="progress-bar">
                                            <div
                                                className="progress-fill high"
                                                style={{
                                                    width: `${pct}%`,
                                                    background: `linear-gradient(90deg, ${categoryColors[category] || '#4361ee'}, ${categoryColors[category] || '#4361ee'}88)`
                                                }}
                                            />
                                        </div>
                                    </div>
                                );
                            })}
                    </div>
                </div>

                {/* Experience & Domain Breakdown */}
                <div className="flex flex-col gap-lg">
                    <div className="card">
                        <div className="card-header">
                            <div className="card-icon accent">📈</div>
                            <div>
                                <div className="card-title">Experience Levels</div>
                                <div className="card-subtitle">User distribution by experience</div>
                            </div>
                        </div>
                        <div className="flex flex-col gap-sm">
                            {stats?.experience_breakdown && Object.entries(stats.experience_breakdown).map(([level, count]) => (
                                <div key={level} className="feature-detail">
                                    <span style={{ fontSize: '1.2rem' }}>
                                        {level === 'beginner' ? '🌱' : level === 'intermediate' ? '🚀' : '⚡'}
                                    </span>
                                    <span style={{ fontWeight: 600, textTransform: 'capitalize', flex: 1 }}>{level}</span>
                                    <span className="badge badge-primary">{count} users</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <div className="card-icon purple">🎯</div>
                            <div>
                                <div className="card-title">Domain Breakdown</div>
                                <div className="card-subtitle">Users grouped by primary domain</div>
                            </div>
                        </div>
                        <div className="flex flex-col gap-sm">
                            {stats?.domain_breakdown && Object.entries(stats.domain_breakdown)
                                .sort(([, a], [, b]) => b - a)
                                .map(([domain, count]) => (
                                    <div key={domain} className="feature-detail">
                                        <span>{domainLabels[domain] || domain}</span>
                                        <span style={{ marginLeft: 'auto' }} className="badge badge-accent">{count}</span>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Recent Users Table */}
            <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                <div className="card-header">
                    <div className="card-icon orange">👥</div>
                    <div>
                        <div className="card-title">Registered Users</div>
                        <div className="card-subtitle">All users in the SkillForge system</div>
                    </div>
                </div>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                {['#', 'Name', 'Skills', 'Experience', 'Domain'].map(h => (
                                    <th key={h} style={{
                                        textAlign: 'left',
                                        padding: '12px 8px',
                                        fontSize: '0.75rem',
                                        fontWeight: 700,
                                        color: 'var(--text-muted)',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.05em'
                                    }}>
                                        {h}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {users.map((user, i) => (
                                <tr
                                    key={user.id}
                                    style={{
                                        borderBottom: '1px solid var(--border-subtle)',
                                        transition: 'background 0.2s',
                                    }}
                                    onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
                                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                                >
                                    <td style={{ padding: '12px 8px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>{i + 1}</td>
                                    <td style={{ padding: '12px 8px', fontWeight: 600, fontSize: '0.9rem' }}>{user.name}</td>
                                    <td style={{ padding: '12px 8px' }}>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                            {user.skills.split(',').slice(0, 4).map((s, j) => (
                                                <span key={j} className="skill-tag" style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
                                                    {s.trim()}
                                                </span>
                                            ))}
                                            {user.skills.split(',').length > 4 && (
                                                <span className="skill-tag" style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
                                                    +{user.skills.split(',').length - 4}
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                    <td style={{ padding: '12px 8px' }}>
                                        <span className={`badge ${user.experience === 'advanced' ? 'badge-accent' : user.experience === 'intermediate' ? 'badge-primary' : 'badge-orange'}`}>
                                            {user.experience}
                                        </span>
                                    </td>
                                    <td style={{ padding: '12px 8px', fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                                        {domainLabels[user.domain] || user.domain}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
