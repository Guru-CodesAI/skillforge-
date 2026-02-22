import { useState } from 'react';

interface RegisterFormProps {
    onAnalyze: (data: {
        name: string;
        skills: string;
        experience: string;
        domain: string;
        interest: string;
    }) => void;
    loading: boolean;
}

const EXPERIENCE_OPTIONS = [
    { value: 'beginner', label: '🌱 Beginner' },
    { value: 'intermediate', label: '🚀 Intermediate' },
    { value: 'advanced', label: '⚡ Advanced' },
];

const DOMAIN_OPTIONS = [
    { value: 'ai_ml', label: '🤖 AI / Machine Learning' },
    { value: 'web_dev', label: '🌐 Web Development' },
    { value: 'mobile', label: '📱 Mobile Development' },
    { value: 'data_science', label: '📊 Data Science' },
    { value: 'ui_ux', label: '🎨 UI/UX Design' },
    { value: 'database', label: '🗄️ Database Engineering' },
    { value: 'devops', label: '⚙️ DevOps / Cloud' },
    { value: 'general', label: '💻 General' },
];

const INTEREST_OPTIONS = [
    { value: 'healthcare', label: '🏥 Healthcare' },
    { value: 'fintech', label: '💰 FinTech' },
    { value: 'edtech', label: '📚 EdTech' },
    { value: 'sustainability', label: '🌿 Sustainability' },
    { value: 'gaming', label: '🎮 Gaming' },
    { value: 'social_good', label: '🤝 Social Good' },
    { value: 'general', label: '🔧 General' },
];

export default function RegisterForm({ onAnalyze, loading }: RegisterFormProps) {
    const [name, setName] = useState('');
    const [skills, setSkills] = useState('');
    const [experience, setExperience] = useState('intermediate');
    const [domain, setDomain] = useState('ai_ml');
    const [interest, setInterest] = useState('healthcare');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!name.trim() || !skills.trim()) return;
        onAnalyze({ name, skills, experience, domain, interest });
    };

    return (
        <form onSubmit={handleSubmit} id="register-form">
            <div className="card" style={{ maxWidth: '680px', margin: '0 auto' }}>
                <div className="card-header">
                    <div className="card-icon primary">⚡</div>
                    <div>
                        <div className="card-title">Register & Find Teammates</div>
                        <div className="card-subtitle">Enter your skills and the AI will find your perfect hackathon team</div>
                    </div>
                </div>

                <div className="form-group">
                    <label htmlFor="name-input" className="form-label">Full Name</label>
                    <input
                        id="name-input"
                        type="text"
                        className="form-input"
                        placeholder="e.g., Arjun Sharma"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="skills-input" className="form-label">Skills</label>
                    <textarea
                        id="skills-input"
                        className="form-input form-textarea"
                        placeholder="e.g., python, react, tensorflow, machine learning, flask"
                        value={skills}
                        onChange={(e) => setSkills(e.target.value)}
                        required
                    />
                    <div className="form-hint">Comma-separated list of your technical skills</div>
                </div>

                <div className="grid grid-3" style={{ marginBottom: 'var(--space-lg)' }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label htmlFor="experience-select" className="form-label">Experience</label>
                        <select
                            id="experience-select"
                            className="form-select"
                            value={experience}
                            onChange={(e) => setExperience(e.target.value)}
                        >
                            {EXPERIENCE_OPTIONS.map(opt => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label htmlFor="domain-select" className="form-label">Domain</label>
                        <select
                            id="domain-select"
                            className="form-select"
                            value={domain}
                            onChange={(e) => setDomain(e.target.value)}
                        >
                            {DOMAIN_OPTIONS.map(opt => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                        </select>
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label htmlFor="interest-select" className="form-label">Interest</label>
                        <select
                            id="interest-select"
                            className="form-select"
                            value={interest}
                            onChange={(e) => setInterest(e.target.value)}
                        >
                            {INTEREST_OPTIONS.map(opt => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <button
                    type="submit"
                    id="analyze-button"
                    className="btn btn-primary btn-lg"
                    disabled={loading || !name.trim() || !skills.trim()}
                    style={{ width: '100%' }}
                >
                    {loading ? (
                        <>
                            <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></span>
                            AI is analyzing...
                        </>
                    ) : (
                        <>🔍 Find My Teammates</>
                    )}
                </button>
            </div>
        </form>
    );
}
