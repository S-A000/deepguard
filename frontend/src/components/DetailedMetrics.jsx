import React, { useEffect, useState } from 'react';

const DetailedMetrics = ({ result, loading }) => {
  // Generate random bars for Audio Spectrum to keep the animation alive
  const [bars, setBars] = useState([]);
  
  useEffect(() => {
    // Only randomize heights, keep colors consistent
    const generateBars = () => {
        const newBars = Array.from({ length: 40 }).map((_, i) => ({
        height: Math.random() * 80 + 10,
        color: i > 25 && i < 35 ? 'bg-red-500' : (i % 3 === 0 ? 'bg-cyan-500' : 'bg-purple-500')
        }));
        setBars(newBars);
    };
    generateBars();
    const interval = setInterval(generateBars, 600);
    return () => clearInterval(interval);
  }, []);

  // Safe extraction
  const scores = result?.branch_scores || {};
  const isPending = !result;

  return (
    <div className="flex flex-col gap-4 flex-1">
      {/* Top Metrics Box with explicit Min Height and dynamic flex to prevent overlap */}
      <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-6 shadow-lg relative flex flex-col justify-between flex-1 min-h-[300px]">
        <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-cyan-500/10 to-transparent pointer-events-none rounded-t-2xl"></div>

        <div className="flex justify-between items-center mb-4 relative z-10 shrink-0">
          <h2 className="text-lg font-semibold text-white">Forensic Details</h2>
          <button className="text-gray-500 hover:text-white shrink-0">...</button>
        </div>

        {/* 4 Branch Scores Bar Chart */}
        <div className={`relative z-10 flex-1 flex flex-col justify-end transition-opacity ${isPending ? 'opacity-40' : 'opacity-100'} min-h-[120px]`}>
          <div className="flex justify-around items-end h-[80%] gap-4 px-2 w-full">
             <Bar label="Spatial" height={scores.spatial || 10} percentage={isPending ? '--%' : `${scores.spatial}%`} color="bg-cyan-400" />
             <Bar label="Physics" height={scores.physics || 10} percentage={isPending ? '--%' : `${scores.physics}%`} color="bg-cyan-400" />
             <Bar label="Forensics" height={scores.forensics || 10} percentage={isPending ? '--%' : `${scores.forensics}%`} color="bg-gradient-to-t from-cyan-500 to-purple-400" />
             <Bar label="Audio" height={scores.audio || 10} percentage={isPending ? '--%' : `${scores.audio}%`} color="bg-gradient-to-t from-cyan-500 to-blue-400" />
          </div>
        </div>

        {/* Separator */}
        <div className="w-full h-px bg-gradient-to-r from-transparent via-[#1f2937] to-transparent my-4 relative z-10 shrink-0"></div>

        {/* Audio Spectrum Anomalies - Flex constraints applied so text doesn't overflow */}
        <div className="relative z-10 shrink-0 flex flex-col justify-end">
          <h3 className="text-gray-300 text-xs mb-2 font-medium">Audio Spectrum Analysis</h3>
          
          <div className={`flex items-end justify-center gap-[2px] h-14 mb-3 transition-opacity ${isPending ? 'opacity-30' : 'opacity-100'}`}>
             {bars.map((bar, i) => (
                <div key={i} className={`w-1 rounded-sm ${bar.color} transition-all duration-300`} style={{ height: `${isPending ? 20 : bar.height}%` }}></div>
             ))}
          </div>
          
          <div className="flex justify-between items-center bg-[#0a0f1c] px-3 py-2 rounded-lg border border-gray-800 flex-wrap gap-2">
             <div className="flex items-center gap-2">
                 <div className={`w-2 h-2 rounded-full ${isPending ? 'bg-gray-600' : 'bg-red-500 animate-pulse'}`}></div>
                 <span className="text-gray-400 text-[11px] font-medium">{isPending ? 'AWAITING SCAN' : 'Anomalies Detected'}</span>
             </div>
             <div className="flex items-center gap-4 ml-auto">
                 <div className="flex flex-col items-end">
                     <p className="text-gray-600 text-[10px] uppercase font-bold tracking-wider">Deviation</p>
                     <p className="text-cyan-400 font-bold text-xs">{isPending ? '--' : scores.audio ? `${scores.audio}%` : '--'}</p>
                 </div>
             </div>
          </div>
        </div>
      </div>

      {/* Analysis Progress Box */}
      <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-5 shadow-lg relative shrink-0 mt-auto">
        <h3 className="text-gray-300 text-sm mb-3 font-medium">Analysis Progress</h3>
        <div className="w-full h-2 bg-[#0a0f1c] rounded-full overflow-hidden mb-2">
            <div className={`h-full rounded-full transition-all duration-1000 ${loading ? 'w-2/3 bg-blue-500 animate-pulse' : (isPending ? 'w-0' : 'w-full bg-gradient-to-r from-green-500 to-cyan-400')} shadow-[0_0_10px_rgba(6,182,212,0.8)]`}></div>
        </div>
        <p className={`text-xs ${loading ? 'text-blue-400' : 'text-gray-500'}`}>{loading ? 'Extracting biometric tensors...' : (isPending ? 'Standby' : 'Scan Complete')}</p>
      </div>
    </div>
  );
};

// Adjusted Bar Component to use a percentage strings directly instead of arbitrary height calculations
const Bar = ({ label, height, percentage, color }) => (
  <div className="flex flex-col items-center flex-1 h-full max-w-[45px]">
     <div className="w-full bg-[#0a0f1c] border border-[#1f2937]/50 rounded-md flex items-end justify-center overflow-visible relative group h-full min-h-[50px]">
        {/* Value Tag */}
        <span className="absolute -top-6 transform transition-all text-white text-[11px] font-bold z-10 drop-shadow-md">
            {percentage}
        </span>
        {/* Bar Fill */}
        <div className={`w-[80%] ${color} rounded-sm shadow-[0_5px_15px_rgba(6,182,212,0.4)] transition-all duration-1000`} style={{ height: `${height}%`, minHeight: '5%' }}></div>
     </div>
     <p className="text-gray-400 text-[10px] mt-2 text-center uppercase font-bold tracking-wider hidden sm:block">{label}</p>
  </div>
);

export default DetailedMetrics;
