import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const UserLogin = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const res = await axios.post('http://127.0.0.1:8000/api/login', { email, password });
            
            // SYNCING WITH SIDEBAR
            localStorage.setItem('role', res.data.role); // Backend se 'admin' ya 'operator' aayega
            localStorage.setItem('userName', res.data.full_name);
            localStorage.setItem('user_id', res.data.user_id);

            navigate('/dashboard');
        } catch (err) {
            alert("Login Failed: Identity not verified.");
        }
    };

    return (
        <div style={styles.container}>
            <div style={styles.card}>
                <h2 style={{color: '#38bdf8'}}>👤 Investigator Login</h2>
                <form onSubmit={handleLogin}>
                    <input style={styles.input} type="email" placeholder="Email" onChange={e => setEmail(e.target.value)} required />
                    <input style={styles.input} type="password" placeholder="Password" onChange={e => setPassword(e.target.value)} required />
                    <button type="submit" style={styles.btn}>Access System</button>
                </form>
                <p style={{marginTop: '15px', color: '#94a3b8'}}>New? <span onClick={() => navigate('/register')} style={{color: '#38bdf8', cursor: 'pointer'}}>Register</span></p>
            </div>
        </div>
    );
};

const styles = {
    container: { height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#020617' },
    card: { backgroundColor: '#1e293b', padding: '40px', borderRadius: '24px', textAlign: 'center', width: '380px', border: '1px solid #38bdf8' },
    input: { width: '100%', padding: '14px', marginBottom: '15px', borderRadius: '10px', backgroundColor: '#0f172a', border: '1px solid #334155', color: 'white', boxSizing: 'border-box' },
    btn: { width: '100%', padding: '14px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '10px', cursor: 'pointer', fontWeight: 'bold' }
};

export default UserLogin;