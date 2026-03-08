import React from 'react';

const VideoUploader = ({ file, setFile, setPreviewUrl, setResult }) => {
    return (
        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px', textAlign: 'center' }}>
            <div style={{ width: '100%' }}>
                <div style={{ fontSize: "50px", textShadow: "0 0 20px #38bdf8", marginBottom: '15px' }}>🧬</div>
                <h3 style={{ color: "#f8fafc", margin: "0 0 10px 0", letterSpacing: "1px", fontSize: '16px' }}>
                    {file ? file.name : "Inject Evidence"}
                </h3>
                <p style={{ color: "#64748b", fontSize: "12px", marginBottom: "25px" }}>MP4, AVI, MOV (Max 50MB)</p>
                <label style={styles.fileButton}>
                    SELECT SOURCE
                    <input 
                        type="file" 
                        onChange={(e) => {
                            if (e.target.files[0]) {
                                setFile(e.target.files[0]);
                                setPreviewUrl(URL.createObjectURL(e.target.files[0]));
                                setResult(null);
                            }
                        }} 
                        style={{ display: "none" }} 
                        accept="video/*" 
                    />
                </label>
            </div>
        </div>
    );
};

const styles = {
    fileButton: {
        background: "linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%)", color: "#001018",
        padding: "12px 30px", borderRadius: "8px", fontWeight: "900", fontSize: "12px",
        cursor: "pointer", display: "inline-block", letterSpacing: "1px", boxShadow: "0 0 15px rgba(56,189,248,0.4)"
    }
};

export default VideoUploader;