import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000,
});

// Types
export interface User {
    id: number;
    name: string;
    skills: string;
    experience: string;
    domain: string;
    interest: string;
    created_at?: string;
}

export interface FeatureDetails {
    skill_similarity: number;
    experience_difference: number;
    domain_match: boolean;
    interest_match: boolean;
}

export interface RecommendedTeammate {
    id: number;
    name: string;
    skills: string;
    experience: string;
    domain: string;
    interest: string;
    similarity_score: number;
    compatibility_score: number;
    feature_details: FeatureDetails;
}

export interface RoleCoverage {
    covered: boolean;
    matching_skills: string[];
    coverage_percentage: number;
    skill_count: number;
}

export interface AnalysisResult {
    success: boolean;
    recommended_teammates: RecommendedTeammate[];
    missing_skills: string[];
    team_strength_level: string;
    coverage_score: number;
    role_coverage: Record<string, RoleCoverage>;
    recommendations: string[];
    analyzed_user: User;
    ai_metadata: {
        tfidf_features: number;
        model_type: string;
        total_users_analyzed: number;
    };
}

export interface TeamBalance {
    success: boolean;
    covered_roles: string[];
    missing_roles: string[];
    role_coverage: Record<string, RoleCoverage>;
    team_strength_level: string;
    coverage_score: number;
    recommendations: string[];
    team_members: { id: number; name: string; skills: string }[];
    total_unique_skills: number;
}

export interface Stats {
    success: boolean;
    total_users: number;
    skills_distribution: Record<string, number>;
    experience_breakdown: Record<string, number>;
    domain_breakdown: Record<string, number>;
}

// API Functions
export const registerUser = async (userData: {
    name: string;
    skills: string;
    experience: string;
    domain: string;
    interest: string;
}) => {
    const response = await api.post('/register', userData);
    return response.data;
};

export const analyzeUser = async (data: { user_id?: number } | {
    name: string;
    skills: string;
    experience: string;
    domain: string;
    interest: string;
}): Promise<AnalysisResult> => {
    const response = await api.post('/analyze', data);
    return response.data;
};

export const getRecommendations = async (userId: number) => {
    const response = await api.get(`/recommendations?user_id=${userId}`);
    return response.data;
};

export const getTeamBalance = async (userIds?: number[]): Promise<TeamBalance> => {
    const params = userIds ? `?user_ids=${userIds.join(',')}` : '';
    const response = await api.get(`/team-balance${params}`);
    return response.data;
};

export const getUsers = async (): Promise<{ success: boolean; users: User[]; count: number }> => {
    const response = await api.get('/users');
    return response.data;
};

export const getStats = async (): Promise<Stats> => {
    const response = await api.get('/stats');
    return response.data;
};

export const healthCheck = async () => {
    const response = await api.get('/health');
    return response.data;
};

export default api;
