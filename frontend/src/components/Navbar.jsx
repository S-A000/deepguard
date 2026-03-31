import React from 'react';

const Navbar = () => {
    return (
        <nav className="h-[70px] bg-[#0a0f1c] border-b border-[#1f2937] flex justify-between items-center px-8 z-50 sticky top-0 transition-all font-sans">
            <div className="flex items-center gap-4">
                <div className="bg-[#1f2937] p-2 rounded-lg border border-gray-700 shadow-sm">
                   <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                </div>
                <h2 className="text-gray-200 font-semibold tracking-wide m-0 mt-0.5">Enterprise System <span className="text-[10px] text-gray-500 font-medium ml-2 uppercase tracking-widest border border-[#1f2937] bg-[#0a0a0f] px-2 py-0.5 rounded-full">v2.1.0</span></h2>
            </div>
            
            <div className="flex items-center gap-6 text-sm">
                <div className="flex items-center gap-2 bg-[#0f172a] px-3 py-1.5 rounded-full border border-[#1f2937] shadow-inner">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                    </span>
                    <span className="text-gray-400 text-xs font-medium uppercase tracking-widest">Online</span>
                </div>
                
                <div className="flex gap-4 text-gray-400 items-center">
                    <div className="hidden sm:flex flex-col items-end">
                       <span className="text-[10px] uppercase font-bold tracking-widest text-[#a855f7]">Mode</span>
                       <span className="text-gray-200 text-xs font-medium">Enterprise AI</span>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;