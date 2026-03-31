import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const AdminLogin = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        
        // --- 🛡️ ADMIN AUTHORIZATION LOGIC ---
        // Email: admin@deepguard.com | Pass: admin123
        if (email === 'admin@deepguard.com' && password === 'admin123') {
            
            // Yahan key ka naam 'role' hona zaroori hai taake Sidebar ise pehchan sakay
            localStorage.setItem('role', 'admin'); 
            localStorage.setItem('userName', 'Super Admin');
            localStorage.setItem('user_id', '1'); // Default Admin ID
            
            console.log("Admin Access Granted. Role set to: admin");
            navigate('/dashboard');
        } else {
            alert("⚠️ ACCESS DENIED: Unauthorized Admin Credentials");
        }
    };

    return (
        <div style={styles.fullPage}>
            <div style={styles.loginCard}>
                <div style={styles.iconCircle}>🛡️</div>
                <h2 style={{ color: '#ef4444', marginBottom: '10px' }}>Admin Command Center</h2>
                <p style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '25px' }}>High-Level Authorization Required</p>
                
                <form onSubmit={handleLogin}>
                    <input 
                        type="email" 
                        placeholder="Admin Email" 
                        style={styles.input}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                    <input 
                        type="password" 
                        placeholder="Security Key" 
                        style={styles.input}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <button type="submit" style={styles.adminBtn}>Verify & Authorize</button>
                </form>
                
                <p onClick={() => navigate('/')} style={styles.backLink}>← Return to Gateway</p>
            </div>
        </div>
    );
};

const styles = {
    fullPage: { height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#020617' },
    loginCard: { backgroundColor: '#1e293b', padding: '40px', borderRadius: '24px', textAlign: 'center', width: '380px', border: '1px solid #ef4444', boxShadow: '0 0 40px rgba(239, 68, 68, 0.15)' },
    iconCircle: { fontSize: '50px', marginBottom: '15px' },
    input: { width: '100%', padding: '14px', marginBottom: '15px', borderRadius: '10px', border: '1px solid #334155', backgroundColor: '#0f172a', color: 'white', outline: 'none', boxSizing: 'border-box' },
    adminBtn: { width: '100%', padding: '14px', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '10px', cursor: 'pointer', fontWeight: 'bold', fontSize: '16px', marginTop: '10px' },
    backLink: { color: '#64748b', marginTop: '25px', cursor: 'pointer', fontSize: '14px', display: 'block', textDecoration: 'underline' }
};

export default AdminLogin;