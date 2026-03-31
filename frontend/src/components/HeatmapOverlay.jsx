import React from 'react';

const HeatmapOverlay = ({ previewUrl, isScanning, verdict }) => {
    return (
        <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
            {isScanning && <div style={styles.scannerLine}></div>}
            {verdict === "FAKE" && <div style={styles.fakeOverlay}></div>}
            
            <div style={styles.header}>EVIDENCE_PLAYBACK</div>
            
            <video src={previewUrl} autoPlay loop muted style={{ width: "100%", height: "100%", objectFit: "cover", opacity: 0.7 }} />
            
            <div style={styles.footer}>
                {isScanning ? "ANALYZING_FRAMES..." : `STATUS: ${verdict || "STANDBY"}`}
            </div>

            <style>{`
                @keyframes scanAnim { 0% { top: 0; } 100% { top: 100%; } }
                @keyframes pulseRed { 0% { opacity: 0; } 50% { opacity: 0.5; } 100% { opacity: 0; } }
            `}</style>
        </div>
    );
};

const styles = {
    header: { position: "absolute", top: "15px", left: "15px", color: "#38bdf8", fontSize: "10px", letterSpacing: "2px", zIndex: 10 },
    scannerLine: { position: "absolute", width: "100%", height: "3px", background: "#38bdf8", boxShadow: "0 0 20px #38bdf8", zIndex: 10, animation: "scanAnim 2s linear infinite" },
    fakeOverlay: { position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: "radial-gradient(circle, transparent 40%, rgba(239, 68, 68, 0.6) 100%)", animation: "pulseRed 1.5s infinite", pointerEvents: "none", zIndex: 5 },
    footer: { position: "absolute", bottom: "15px", left: "15px", color: "#64748b", fontSize: "10px", fontFamily: "monospace", zIndex: 10, background: 'rgba(0,0,0,0.5)', padding: '5px 10px', borderRadius: '4px' }
};

export default HeatmapOverlay;