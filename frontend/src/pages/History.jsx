import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:8000/api/history');
                setHistory(response.data);
                setLoading(false);
            } catch (error) {
                console.error("Database fetch error:", error);
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    return (
        <div style={styles.pageWrapper}>
            {/* 1. Fixed Sidebar - Yahan se Dashboard wapis ja saktay hain */}
            <Sidebar />

            {/* 2. Main Content Area */}
            <div style={styles.mainArea}>
                <Navbar />

                <div style={styles.contentPadding}>
                    <header style={styles.headerSection}>
                        <h2 style={{ color: '#38bdf8', margin: 0, textTransform: 'uppercase', letterSpacing: '2px' }}>
                            📂 Global Governance Logs
                        </h2>
                        <p style={{ color: '#94a3b8', fontSize: '14px' }}>System-wide forensic analysis records</p>
                    </header>
                    
                    {loading ? (
                        <div style={{ textAlign: 'center', color: '#38bdf8', marginTop: '100px' }}>
                            <div className="spinner"></div>
                            <p>Accessing SQL Server Records...</p>
                        </div>
                    ) : (
                        <div style={styles.tableContainer}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', color: '#cbd5e1' }}>
                                <thead>
                                    <tr style={{ backgroundColor: '#0f172a', textAlign: 'left' }}>
                                        <th style={styles.th}>Source File</th>
                                        <th style={styles.th}>AI Verdict</th>
                                        <th style={styles.th}>Confidence</th>
                                        <th style={styles.th}>Processing Time</th>
                                        <th style={styles.th}>Timestamp</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {history.length > 0 ? history.map((item, index) => (
                                        <tr key={item.analysis_id} style={{ 
                                            borderBottom: '1px solid #334155',
                                            backgroundColor: index % 2 === 0 ? 'transparent' : 'rgba(15, 23, 42, 0.3)'
                                        }} className="table-row">
                                            <td style={styles.td}>💾 {item.filename}</td>
                                            <td style={{ 
                                                ...styles.td, 
                                                color: item.verdict === "FAKE" ? "#f87171" : "#4ade80",
                                                fontWeight: 'bold'
                                            }}>
                                                {item.verdict}
                                            </td>
                                            <td style={styles.td}>{item.confidence_score}%</td>
                                            <td style={styles.td}>{item.processing_time_sec}s</td>
                                            <td style={styles.td}>
                                                {new Date(item.timestamp).toLocaleString()}
                                            </td>
                                        </tr>
                                    )) : (
                                        <tr>
                                            <td colSpan="5" style={{textAlign: 'center', padding: '40px', color: '#64748b'}}>No records found in governance database.</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>

            <style>{`
                .spinner {
                    width: 40px; height: 40px; border: 4px solid #1e293b;
                    border-top: 4px solid #38bdf8; border-radius: 50%;
                    animation: spin 1s linear infinite; margin: 0 auto 20px;
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                .table-row { transition: 0.3s; }
                .table-row:hover { background-color: rgba(56, 189, 248, 0.1) !important; cursor: default; }
            `}</style>
        </div>
    );
};

const styles = {
    pageWrapper: { display: 'flex', height: '100vh', backgroundColor: '#020617', overflow: 'hidden' },
    mainArea: { 
        flex: 1, 
        marginLeft: '260px', 
        display: 'flex', 
        flexDirection: 'column', 
        overflowY: 'auto' 
    },
    contentPadding: { padding: '40px' },
    headerSection: { marginBottom: '30px' },
    tableContainer: { 
        overflow: 'hidden', borderRadius: '15px', border: '1px solid #334155',
        background: 'rgba(30, 41, 59, 0.7)', backdropFilter: 'blur(10px)',
        boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
    },
    th: { padding: '18px 20px', color: '#38bdf8', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px' },
    td: { padding: '18px 20px', fontSize: '14px' }
};

export default History;