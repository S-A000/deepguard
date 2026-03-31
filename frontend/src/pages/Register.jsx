import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

const Register = () => {
    const [formData, setFormData] = useState({ full_name: '', email: '', password: '' });
    const navigate = useNavigate();

    const handleRegister = async (e) => {
        e.preventDefault();
        try {
            await axios.post('http://127.0.0.1:8000/api/signup', formData);
            alert("Registration Successful! Please Login.");
            navigate('/login-user'); // Login page par bhej do
        } catch (err) {
            alert(err.response?.data?.detail || "Registration Failed");
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.card}>
                <h2 style={{color: '#38bdf8'}}>🛡️ Create Identity</h2>
                <p style={{color: '#94a3b8'}}>Join DeepGuard Governance Network</p>
                <form onSubmit={handleRegister} style={{marginTop: '20px'}}>
                    <input style={styles.input} type="text" placeholder="Full Name" onChange={(e) => setFormData({...formData, full_name: e.target.value})} required />
                    <input style={styles.input} type="email" placeholder="Email Address" onChange={(e) => setFormData({...formData, email: e.target.value})} required />
                    <input style={styles.input} type="password" placeholder="Create Password" onChange={(e) => setFormData({...formData, password: e.target.value})} required />
                    <button type="submit" style={styles.btn}>Initialize Account</button>
                </form>
                <p style={{marginTop: '20px', fontSize: '14px'}}>
                    Already have an account? <Link to="/login-user" style={{color: '#38bdf8'}}>Login here</Link>
                </p>
            </div>
        </div>
    );
};

const styles = {
    container: { height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#020617' },
    card: { backgroundColor: '#1e293b', padding: '40px', borderRadius: '20px', textAlign: 'center', width: '380px', border: '1px solid #38bdf8' },
    input: { width: '100%', padding: '12px', marginBottom: '15px', borderRadius: '8px', border: '1px solid #334155', backgroundColor: '#0f172a', color: 'white', boxSizing: 'border-box' },
    btn: { width: '100%', padding: '12px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }
};

export default Register;