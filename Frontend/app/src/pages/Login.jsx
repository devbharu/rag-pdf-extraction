import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        const success = await login(username, password);
        if (success) {
            navigate('/');
        } else {
            setError('Invalid username or password');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-stone-50 text-stone-800 font-sans">
            <div className="bg-white p-6 rounded-xl shadow-sm w-full max-w-sm border border-stone-200">
                <div className="flex flex-col items-center mb-6">
                    <img src="/logo.png" className="w-12 h-12 mb-3" alt="Logo" />
                    <h2 className="text-2xl font-semibold text-center text-stone-800 tracking-tight">Welcome Back</h2>
                </div>
                {error && <p className="text-red-500 text-center mb-4 text-xs font-medium bg-red-50 py-1 rounded">{error}</p>}
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-xs font-medium text-stone-500 mb-1.5 uppercase tracking-wide">Username</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full px-3 py-2.5 rounded-lg bg-stone-50 border border-stone-200 focus:border-stone-400 focus:ring-1 focus:ring-stone-400 outline-none transition-all text-sm text-stone-800 placeholder:text-stone-400"
                            placeholder="Enter your username"
                            required
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-stone-500 mb-1.5 uppercase tracking-wide">Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-3 py-2.5 rounded-lg bg-stone-50 border border-stone-200 focus:border-stone-400 focus:ring-1 focus:ring-stone-400 outline-none transition-all text-sm text-stone-800 placeholder:text-stone-400"
                            placeholder="Enter your password"
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full py-2.5 px-4 bg-stone-900 hover:bg-stone-800 text-white rounded-lg text-sm font-medium shadow-sm transition-all hover:shadow active:scale-[0.98]"
                    >
                        Sign In
                    </button>
                </form>
                <p className="mt-6 text-center text-stone-400 text-xs">
                    Don't have an account?{' '}
                    <Link to="/register" className="text-stone-800 hover:text-teal-700 font-semibold transition-colors">
                        Sign up
                    </Link>
                </p>
            </div>
        </div>
    );
};

export default Login;
