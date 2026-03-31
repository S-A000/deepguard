import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from '../components/Sidebar';
import Navbar from '../components/Navbar';

const UserManagement = () => {
    const [users, setUsers] = useState([]);

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/api/admin/users').then(res => setUsers(res.data));
    }, []);

    const handleDelete = async (id) => {
        if (window.confirm("Revoke this user's access permanently?")) {
            await axios.delete(`http://127.0.0.1:8000/api/admin/delete-user/${id}`);
            setUsers(users.filter(u => u.user_id !== id));
        }
    };

    return (
        <div style={{ display: 'flex', backgroundColor: '#020617', minHeight: '100vh' }}>
            <Sidebar />
            <div style={{ flex: 1, marginLeft: '260px' }}>
                <Navbar />
                <div style={{ padding: '40px' }}>
                    <div style={styles.header}>
                        <h2 style={{ color: '#f8fafc', margin: 0 }}>Identity Network</h2>
                        <p style={{ color: '#94a3b8' }}>Manage authorized forensic investigators</p>
                    </div>

                    <div style={styles.tableCard}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    <th style={styles.th}>INVESTIGATOR</th>
                                    <th style={styles.th}>EMAIL</th>
                                    <th style={styles.th}>ROLE</th>
                                    <th style={styles.th}>STATUS</th>
                                    <th style={styles.th}>ACTIONS</th>
                                </tr>
                            </thead>
                            <tbody>
                                {users.map(u => (
                                    <tr key={u.user_id} style={styles.tr}>
                                        <td style={styles.td}><b>{u.full_name}</b></td>
                                        <td style={styles.td}>{u.email}</td>
                                        <td style={styles.td}><span style={styles.roleTag(u.role)}>{u.role.toUpperCase()}</span></td>
                                        <td style={styles.td}><span style={styles.statusDot}></span> Active</td>
                                        <td style={styles.td}>
                                            <button onClick={() => handleDelete(u.user_id)} style={styles.delBtn}>REVOKE</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

const styles = {
    header: { marginBottom: '30px' },
    tableCard: { backgroundColor: '#0f172a', borderRadius: '20px', border: '1px solid #1e293b', padding: '20px', overflow: 'hidden' },
    table: { width: '100%', borderCollapse: 'collapse', color: '#cbd5e1' },
    th: { textAlign: 'left', padding: '15px', color: '#64748b', fontSize: '12px', letterSpacing: '1px', borderBottom: '1px solid #1e293b' },
    td: { padding: '18px 15px', borderBottom: '1px solid #1e293b55' },
    tr: { transition: '0.3s', ':hover': { backgroundColor: '#1e293b' } },
    roleTag: (role) => ({ padding: '4px 10px', borderRadius: '6px', fontSize: '10px', fontWeight: 'bold', backgroundColor: role === 'admin' ? '#38bdf822' : '#1e293b', color: role === 'admin' ? '#38bdf8' : '#94a3b8', border: `1px solid ${role === 'admin' ? '#38bdf855' : '#334155'}` }),
    statusDot: { display: 'inline-block', width: '8px', height: '8px', backgroundColor: '#22c55e', borderRadius: '50%', marginRight: '8px' },
    delBtn: { backgroundColor: 'transparent', color: '#ef4444', border: '1px solid #ef444455', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '11px', fontWeight: 'bold', transition: '0.3s' }
};

export default UserManagement;