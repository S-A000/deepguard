import React, { useState, useEffect } from "react";
import axios from "axios";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

// Aapke alag files wale components
import VideoUploader from "../components/VideoUploader";
import ResultGraph from "../components/ResultGraph";
import HeatmapOverlay from "../components/HeatmapOverlay";

const Dashboard = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [userName, setUserName] = useState("Investigator");

    useEffect(() => {
        const savedName = localStorage.getItem("userName");
        if (savedName) setUserName(savedName);
    }, []);

    const handleUpload = async () => {
        if (!file) return alert("⚠️ Select evidence first!");
        
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("user_id", localStorage.getItem("user_id") || 1);

        try {
            const response = await axios.post("http://127.0.0.1:8000/api/analyze", formData);
            setResult(response.data);
        } catch (error) {
            alert("❌ System Error: Forensic Engine Offline.");
        } finally {
            setLoading(false);
        }
    };

    // Dummy data taake graph hamesha bhara hua nazar aaye
    const dummyScores = { spatial: 88, physics: 45, forensics: 72, audio: 91 };

    return (
        <div style={styles.pageWrapper}>
            <Sidebar />
            <div style={styles.mainArea}>
                <Navbar />

                <div style={styles.contentPadding}>
                    <header style={styles.welcomeHeader}>
                        <div>
                            <h1 style={styles.mainTitle}>Forensic Intelligence Console</h1>
                            <p style={styles.subTitle}>Active Subject: <span style={{ color: "#38bdf8" }}>{userName}</span></p>
                        </div>
                        <div style={styles.liveBadge}>● ENGINE ONLINE</div>
                    </header>

                    <div style={styles.contentGrid}>
                        
                        {/* LEFT COLUMN: Uploader & Heatmap */}
                        <div style={styles.inputColumn}>
                            
                            {/* 1. Uploader (Hamesha dikhega) */}
                            <div style={styles.videoBox}>
                                <VideoUploader file={file} setFile={setFile} setPreviewUrl={setPreviewUrl} setResult={setResult} />
                            </div>

                            {/* 2. Heatmap Overlay (Hamesha dikhega test karne ke liye) */}
                            <div style={{...styles.videoBox, marginTop: '20px'}}>
                                <HeatmapOverlay 
                                    previewUrl={previewUrl || "https://www.w3schools.com/html/mov_bbb.mp4"} 
                                    isScanning={loading} 
                                    verdict={result ? result.verdict : "FAKE"} 
                                />
                            </div>

                            <button onClick={handleUpload} disabled={loading} style={styles.scanBtn(loading)}>
                                {loading ? "EXTRACTING SIGNALS..." : "INITIATE DEEP SCAN"}
                            </button>
                        </div>

                        {/* RIGHT COLUMN: Results & Graphs (Hamesha dikhega) */}
                        <div style={styles.resultsColumn}>
                            <div style={styles.verdictPanel(result ? result.verdict : "FAKE")}>
                                <h1 style={{margin: 0, fontSize: '3.5rem', fontWeight: 900, letterSpacing: '-2px'}}>
                                    {result ? result.verdict : "FAKE"}
                                </h1>
                                <p style={{letterSpacing: '2px', fontSize: '13px', opacity: 0.8, marginTop: '10px'}}>
                                    CONFIDENCE INDEX: <b style={{fontSize: '16px', color: '#fff'}}>{result ? result.confidence : "94.5"}%</b>
                                </p>
                            </div>

                            <div style={{ marginTop: "25px" }}>
                                {/* 3. Result Graph (Hamesha dikhega) */}
                                <ResultGraph scores={result ? result.branch_scores : dummyScores} />
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            <style>{`
                body::before {
                    content: ""; position: fixed; width: 100%; height: 100%;
                    background-image: linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
                    background-size: 50px 50px; pointer-events: none; z-index: 0;
                }
            `}</style>
        </div>
    );
};

const styles = {
    pageWrapper: { display: "flex", height: "100vh", backgroundColor: "#000814", overflow: "hidden", fontFamily: "'Inter', sans-serif" },
    mainArea: { flex: 1, marginLeft: "260px", display: "flex", flexDirection: "column", overflowY: "auto", background: "radial-gradient(circle at 50% 0%, rgba(15,23,42,1) 0%, rgba(0,8,20,1) 100%)" },
    contentPadding: { padding: "40px 50px", maxWidth: "1300px", margin: "0 auto", width: "100%" },
    
    welcomeHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "40px" },
    mainTitle: { margin: 0, color: "#f8fafc", fontSize: "2rem", fontWeight: "800", letterSpacing: "-1px" },
    subTitle: { color: "#64748b", fontSize: "13px", marginTop: "5px" },
    liveBadge: { padding: "8px 15px", backgroundColor: "rgba(16, 185, 129, 0.1)", color: "#10b981", borderRadius: "20px", fontSize: "11px", fontWeight: "bold", border: "1px solid rgba(16, 185, 129, 0.3)" },
    
    // Grid Setup
    contentGrid: {
        display: "grid",
        gridTemplateColumns: "400px 1fr", 
        gap: "40px",
        alignItems: "start"
    },

    inputColumn: { display: "flex", flexDirection: "column", gap: "10px" },
    
    videoBox: { 
        width: "100%", height: "300px", borderRadius: "20px", overflow: "hidden",
        backgroundColor: "rgba(15, 23, 42, 0.5)", border: "1px solid rgba(56, 189, 248, 0.15)",
        boxShadow: "0 15px 35px rgba(0,0,0,0.5)", position: 'relative'
    },

    resultsColumn: { display: "flex", flexDirection: "column", gap: "25px" },

    scanBtn: (disabled) => ({
        width: "100%", padding: "18px", borderRadius: "14px", border: disabled ? "1px solid #1e293b" : "none",
        fontSize: "14px", fontWeight: "900", letterSpacing: "2px",
        background: disabled ? "#0f172a" : "linear-gradient(90deg, #38bdf8 0%, #2563eb 100%)",
        color: disabled ? "#475569" : "#ffffff", cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.3s ease", marginTop: "15px"
    }),

    verdictPanel: (verdict) => ({
        padding: "35px", borderRadius: "24px", textAlign: "center", color: "white",
        background: verdict === "FAKE" ? "linear-gradient(135deg, #7f1d1d, #450a0a)" : "linear-gradient(135deg, #064e3b, #022c22)",
        border: `1px solid ${verdict === "FAKE" ? "#ef4444" : "#10b981"}`,
        boxShadow: verdict === "FAKE" ? "0 20px 40px rgba(239,68,68,0.3)" : "0 20px 40px rgba(16,185,129,0.3)"
    })
};

export default Dashboard;