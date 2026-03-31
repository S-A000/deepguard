import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Navbar from '../components/Navbar';

const AssignMember = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({ full_name: '', email: '', password: '', role: 'operator' });

    const handleCreate = async (e) => {
        e.preventDefault();
        try {
            await axios.post('http://127.0.0.1:8000/api/admin/create-member', formData);
            alert("New Identity Authorized!");
            navigate('/user-management'); // Redirect to table page
        } catch (err) { alert("Authorization failed."); }
    };

    return (
        <div style={{ display: 'flex', backgroundColor: '#020617', minHeight: '100vh' }}>
            <Sidebar />
            <div style={{ flex: 1, marginLeft: '260px' }}>
                <Navbar />
                <div style={styles.formWrapper}>
                    <div style={styles.glassCard}>
                        <div style={styles.iconCircle}>🛡️</div>
                        <h2 style={{ color: '#38bdf8', textAlign: 'center', marginBottom: '10px' }}>Assign Member</h2>
                        <p style={{ color: '#94a3b8', textAlign: 'center', marginBottom: '30px', fontSize: '14px' }}>Authorize new investigator into the DeepGuard network</p>
                        
                        <form onSubmit={handleCreate}>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>FULL IDENTITY NAME</label>
                                <input style={styles.input} placeholder="e.g. John Doe" onChange={e => setFormData({...formData, full_name: e.target.value})} required />
                            </div>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>GOVERNMENT EMAIL</label>
                                <input style={styles.input} type="email" placeholder="name@agency.gov" onChange={e => setFormData({...formData, email: e.target.value})} required />
                            </div>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>SECURITY KEY (PASSWORD)</label>
                                <input style={styles.input} type="password" placeholder="••••••••" onChange={e => setFormData({...formData, password: e.target.value})} required />
                            </div>
                            <div style={styles.inputGroup}>
                                <label style={styles.label}>CLEARANCE LEVEL (ROLE)</label>
                                <select style={styles.input} onChange={e => setFormData({...formData, role: e.target.value})}>
                                    <option value="operator">Forensic Operator</option>
                                    <option value="admin">System Administrator</option>
                                </select>
                            </div>
                            <button type="submit" style={styles.submitBtn}>AUTHORIZE & CREATE</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
};

const styles = {
    formWrapper: { display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 80px)' },
    glassCard: { width: '450px', backgroundColor: 'rgba(30, 41, 59, 0.5)', backdropFilter: 'blur(10px)', padding: '40px', borderRadius: '30px', border: '1px solid rgba(56, 189, 248, 0.2)', boxShadow: '0 25px 50px rgba(0,0,0,0.5)' },
    iconCircle: { width: '60px', height: '60px', backgroundColor: '#0f172a', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', margin: '0 auto 20px', fontSize: '30px', border: '1px solid #38bdf8' },
    inputGroup: { marginBottom: '20px' },
    label: { fontSize: '10px', color: '#64748b', letterSpacing: '1px', marginBottom: '8px', display: 'block' },
    input: { width: '100%', padding: '14px', borderRadius: '12px', backgroundColor: '#0f172a', border: '1px solid #334155', color: 'white', outline: 'none', transition: '0.3s', boxSizing: 'border-box' },
    submitBtn: { width: '100%', padding: '16px', backgroundColor: '#38bdf8', color: '#0f172a', border: 'none', borderRadius: '12px', fontWeight: 'bold', cursor: 'pointer', marginTop: '20px', fontSize: '14px', letterSpacing: '1px' }
};

export default AssignMember;