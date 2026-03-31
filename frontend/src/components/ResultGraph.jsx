import React from 'react';

const ResultGraph = ({ scores }) => {
    const s = scores || { spatial: 85, physics: 30, forensics: 60, audio: 45 }; 
    
    return (
        <div style={styles.graphContainer}>
            <h4 style={styles.graphTitle}>🔬 TECHNICAL SIGNAL ANALYSIS</h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "30px" }}>
                <MetricBar label="Spatial Consistency" score={s.spatial} />
                <MetricBar label="Physical Integrity" score={s.physics} />
                <MetricBar label="Digital Forensics" score={s.forensics} />
                <MetricBar label="Audio Biometrics" score={s.audio} />
            </div>
        </div>
    );
};

const MetricBar = ({ label, score }) => (
    <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "12px", color: "#94a3b8" }}>{label}</span>
            <span style={{ fontSize: "13px", color: "#f8fafc", fontWeight: "bold" }}>{score}%</span>
        </div>
        <div style={{ height: "6px", backgroundColor: "#0f172a", borderRadius: "3px", border: "1px solid #1e293b", overflow: 'hidden' }}>
            <div style={{
                height: "100%", width: `${score}%`,
                background: score > 75 ? "linear-gradient(90deg,#ef4444,#b91c1c)" : "linear-gradient(90deg,#38bdf8,#0ea5e9)",
                transition: "width 1.5s cubic-bezier(.17,.67,.83,.67)",
                boxShadow: score > 75 ? "0 0 10px rgba(239,68,68,0.8)" : "0 0 10px rgba(56,189,248,0.8)"
            }}></div>
        </div>
    </div>
);

const styles = {
    graphContainer: { padding: "35px", background: "rgba(2, 6, 23, 0.7)", borderRadius: "24px", border: "1px solid rgba(56,189,248,0.1)", backdropFilter: "blur(10px)" },
    graphTitle: { color: "#38bdf8", fontSize: "12px", marginBottom: "30px", letterSpacing: "2px", textAlign: "center", marginTop: 0 }
};

export default ResultGraph;