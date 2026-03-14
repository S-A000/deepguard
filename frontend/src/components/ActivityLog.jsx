import React from 'react';

const ActivityLog = () => {
  return (
    <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-5 shadow-lg relative overflow-hidden flex-1 min-h-[160px]">
      <div className="absolute top-0 left-0 w-full h-20 bg-gradient-to-b from-cyan-500/10 to-transparent pointer-events-none rounded-t-2xl"></div>

      <div className="flex justify-between items-center mb-5 relative z-10">
        <h2 className="text-lg font-semibold text-white">Activity Log</h2>
        <button className="text-gray-500 hover:text-white">...</button>
      </div>

      <div className="flex flex-col gap-0 relative z-10 pl-2">
        {/* Connection Line */}
        <div className="absolute left-[20px] top-[14px] bottom-[14px] w-px bg-gray-700"></div>

        <div className="flex items-start gap-4 relative mb-5">
           <div className="w-6 h-6 rounded-full bg-[#1f2937] flex items-center justify-center relative z-10 mt-0.5 ring-4 ring-[#11131a]">
              <svg className="w-3 h-3 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
           </div>
           <div className="flex-1">
              <div className="flex justify-between items-center mb-0.5">
                  <h4 className="text-gray-200 text-sm font-medium">Recent Events</h4>
                  <span className="text-gray-500 text-xs bg-[#0a0f1c] px-2 py-0.5 rounded">2 hour ago</span>
              </div>
              <p className="text-gray-500 text-xs">Deepfake_Sample_01.mp4</p>
           </div>
        </div>

        <div className="flex items-start gap-4 relative opacity-50">
           <div className="w-6 h-6 rounded-full bg-[#1f2937] flex items-center justify-center relative z-10 mt-0.5 ring-4 ring-[#11131a]">
              <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" /></svg>
           </div>
           <div className="flex-1">
              <div className="flex justify-between items-center mb-0.5">
                  <h4 className="text-gray-200 text-sm font-medium">Recent Event</h4>
                  <span className="text-gray-500 text-xs bg-[#0a0f1c] px-2 py-0.5 rounded">2 hour ago</span>
              </div>
              <p className="text-gray-500 text-xs">Deepfake activity Log</p>
           </div>
        </div>

      </div>
    </div>
  );
};

export default ActivityLog;
