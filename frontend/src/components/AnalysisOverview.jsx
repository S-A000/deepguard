import React from 'react';

const AnalysisOverview = ({ result, loading }) => {
  // Use real score or default to 0 for mockup
  const score = result ? result.confidence : 0; 
  const displayScore = loading ? "..." : (result ? score : "--");
  const missingScore = loading ? "..." : (result ? (100 - score).toFixed(0) : "--");

  // Dash array length for semi-circle: pi * r = 3.14159 * 40 ≈ 125.6
  // To fill from left to right, we use stroke-dashoffset
  const arcLength = 125.66;
  
  // Custom formula so high confidence of FAKE correctly fills the RED side if we change the gradient later.
  // For now, higher score = filled gauge
  const offset = loading || !result ? arcLength : arcLength * (1 - score / 100);

  // Dynamic risk calculation
  const getRiskLevel = (val) => {
    if (val >= 80) return { text: "Critical Risk", class: "text-red-500" };
    if (val >= 50) return { text: "Elevated Risk", class: "text-orange-500" };
    return { text: "Authentic / Low Risk", class: "text-green-500" };
  };

  const risk = getRiskLevel(score);

  // Safely extract scores from backend format
  const branchScores = result?.branch_scores || {};
  
  const features = [
    { label: "Spatial Analysis", score: branchScores.spatial, iconPath: "M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" },
    { label: "Physics Validation", score: branchScores.physics, iconPath: "M13 10V3L4 14h7v7l9-11h-7z" },
    { label: "Forensic Integrity", score: branchScores.forensics, iconPath: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" }
  ];

  return (
    <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-6 h-[100%] flex flex-col shadow-lg relative min-h-[450px]">
      <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-cyan-500/5 to-transparent pointer-events-none rounded-t-2xl"></div>

      <div className="flex justify-between items-center mb-6 relative z-10">
        <h2 className="text-lg font-semibold text-white truncate">Analysis Overview</h2>
        <button className="text-gray-500 hover:text-white shrink-0">...</button>
      </div>

      {/* Gauge Chart Area */}
      <div className="flex flex-col items-center justify-center relative mb-8 flex-1">
        <div className="w-64 h-32 relative">
          <svg viewBox="0 0 100 50" className="w-full h-full overflow-visible">
             <defs>
              <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#22c55e" /> {/* Green - Authentic */}
                <stop offset="30%" stopColor="#eab308" /> {/* Yellow - Suspicious */}
                <stop offset="60%" stopColor="#f97316" /> {/* Orange - High Risk */}
                <stop offset="100%" stopColor="#ef4444" /> {/* Red - Deepfake */}
              </linearGradient>
              <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>
            {/* Background Track */}
            <path 
              d="M 10 50 A 40 40 0 0 1 90 50" 
              fill="none" 
              stroke="#1f2937" 
              strokeWidth="10" 
              strokeLinecap="round" 
            />
            {/* Foreground Arc */}
            <path 
              d="M 10 50 A 40 40 0 0 1 90 50" 
              fill="none" 
              stroke="url(#gauge-gradient)" 
              strokeWidth="10" 
              strokeLinecap="round" 
              strokeDasharray={arcLength}
              strokeDashoffset={offset}
              filter="url(#glow)"
              style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)" }}
            />
          </svg>

          {/* Center Text */}
          <div className="absolute inset-x-0 bottom-0 flex flex-col items-center justify-end h-full font-sans">
             <span className="text-[42px] font-bold bg-clip-text text-white tracking-tight leading-none mb-1">
                {displayScore}{result ? '%' : ''}
             </span>
             <span className={`text-xs tracking-widest font-semibold uppercase mb-2 ${result ? (result.verdict === 'FAKE' ? 'text-red-400' : 'text-green-400') : 'text-gray-500'}`}>
                 {result ? result.verdict : 'WAITING'}
             </span>
             {result && (
                 <div className="bg-[#0f172a] border border-gray-700 text-gray-400 text-[10px] py-0.5 px-3 rounded-full flex items-center justify-center font-bold">
                   Margin: {missingScore}%
                 </div>
             )}
          </div>
        </div>
        
        {/* Dynamic Risk Tag */}
        <p className="text-gray-300 text-[13px] mt-6 font-medium">Overall Deepfake Probability</p>
        <p className={`text-xs font-bold uppercase tracking-widest mt-1 ${result ? risk.class : 'text-gray-600'}`}>
            {result ? risk.text : "PENDING SCAN"}
        </p>
      </div>

      {/* Feature List (Dynamic Binding to branch_scores) */}
      <div className="flex flex-col gap-3 z-10 w-full mt-auto">
        {features.map((feat, idx) => {
            const val = feat.score !== undefined ? feat.score : '--';
            const statText = val === '--' ? 'N/A' : (val >= 75 ? 'High Deviation' : (val >= 40 ? 'Moderate' : 'Normal'));
            const color = val === '--' ? 'text-gray-600' : (val >= 75 ? 'text-red-400' : (val >= 40 ? 'text-orange-400' : 'text-green-400'));
            
            return (
                <FeatureRow 
                    key={idx}
                    icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={feat.iconPath} /></svg>}
                    label={feat.label}
                    score={val === '--' ? '--' : `${val}%`}
                    status={statText}
                    colorClass={color}
                    isPending={!result}
                />
            );
        })}
      </div>
    </div>
  );
};

const FeatureRow = ({ icon, label, score, status, colorClass, isPending }) => (
  <div className={`bg-[#0a0f1c] border border-[#1f2937] rounded-xl p-3 flex items-center justify-between transition-opacity ${isPending ? 'opacity-50' : 'opacity-100'}`}>
    <div className="flex items-center gap-3">
      <div className="bg-[#1f2937] p-2 rounded-lg text-gray-400 flex-shrink-0">
        {icon}
      </div>
      <div className="flex flex-col truncate">
        <span className="text-gray-200 text-sm font-medium truncate">{label}</span>
        <span className="text-gray-500 text-[11px] font-medium tracking-wide">
            {isPending ? 'AWAITING DATA' : status.toUpperCase()}
        </span>
      </div>
    </div>
    <div className={`font-bold text-sm tracking-wider flex-shrink-0 ${colorClass}`}>
      {score}
    </div>
  </div>
);

export default AnalysisOverview;
