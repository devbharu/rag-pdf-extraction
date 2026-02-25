import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  
useEffect(() => {
  if (token) {
    // Set default authorization header for all axios requests
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    setUser({ username: 'User' });
  } else {
    // Remove authorization header if no token
    delete axios.defaults.headers.common['Authorization'];
  }
  setLoading(false);
}, [token]);

  const login = async (username, password) => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      const response = await axios.post('http://127.0.0.1:8000/token', formData);
      const { access_token } = response.data;

      localStorage.setItem('token', access_token);
      setToken(access_token);
      setUser({ username });
      return true;
    } catch (error) {
      console.error("Login failed", error);
      return false;
    }
  };

  const register = async (username, password) => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);

      await axios.post('http://127.0.0.1:8000/register', formData);
      return true;
    } catch (error) {
      console.error("Registration failed", error);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
