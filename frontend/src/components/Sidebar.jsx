import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const Sidebar = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const userRole = localStorage.getItem('role');

    const handleLogout = () => {
        localStorage.clear();
        navigate('/');
    };

    return (
        <div className="w-[260px] bg-[#0a0f1c] h-screen fixed left-0 top-0 flex flex-col border-r border-[#1f2937] z-[100] shadow-2xl font-sans">
            <div className="flex-1 px-5 py-8 overflow-y-auto">
                {/* Logo Area */}
                <div className="text-center mb-10 flex flex-col items-center">
                    <div className="bg-gradient-to-br from-cyan-500 to-purple-600 p-0.5 rounded-xl mb-3 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                        <div className="bg-[#0f172a] p-2.5 rounded-xl">
                            <span className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">DD</span>
                        </div>
                    </div>
                    <h3 className="text-white text-lg font-bold tracking-tight m-0">DeepGuard<br/><span className="text-gray-400 font-medium">Enterprise</span></h3>
                    <span className="text-[10px] text-cyan-500 font-bold tracking-[0.2em] mt-2 bg-cyan-500/10 px-2 py-0.5 rounded-full border border-cyan-500/20">
                        {userRole?.toUpperCase() || 'OPERATOR'}
                    </span>
                </div>

                <div className="flex flex-col gap-1">
                    <p className="text-[10px] text-gray-500 tracking-widest font-bold ml-2 mb-2 mt-4">CORE WORKSPACE</p>
                    
                    <Link to="/dashboard" className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${location.pathname === '/dashboard' ? 'bg-gradient-to-r from-cyan-500/10 to-transparent text-cyan-400 border-l-2 border-cyan-400 font-medium' : 'text-gray-400 hover:bg-[#11131a] hover:text-gray-200'}`}>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                        Dashboard
                    </Link>
                    
                    <Link to="/history" className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${location.pathname === '/history' ? 'bg-gradient-to-r from-cyan-500/10 to-transparent text-cyan-400 border-l-2 border-cyan-400 font-medium' : 'text-gray-400 hover:bg-[#11131a] hover:text-gray-200'}`}>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                        Forensic History
                    </Link>

                    {userRole === 'admin' && (
                        <>
                            <p className="text-[10px] text-gray-500 tracking-widest font-bold ml-2 mb-2 mt-6">IDENTITY MANAGEMENT</p>
                            <Link to="/assign-member" className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${location.pathname === '/assign-member' ? 'bg-gradient-to-r from-purple-500/10 to-transparent text-purple-400 border-l-2 border-purple-400 font-medium' : 'text-gray-400 hover:bg-[#11131a] hover:text-gray-200'}`}>
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"></path></svg>
                                Assign Member
                            </Link>
                            <Link to="/user-management" className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${location.pathname === '/user-management' ? 'bg-gradient-to-r from-purple-500/10 to-transparent text-purple-400 border-l-2 border-purple-400 font-medium' : 'text-gray-400 hover:bg-[#11131a] hover:text-gray-200'}`}>
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
                                User Management
                            </Link>
                        </>
                    )}
                </div>
            </div>

            <div className="p-5 border-t border-[#1f2937] bg-[#0a0f1c]">
                <button onClick={handleLogout} className="flex items-center justify-center gap-2 w-full py-3 bg-red-500/5 hover:bg-red-500/10 text-red-500 border border-red-500/20 rounded-xl font-bold tracking-wider text-xs transition-all">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
                    EXIT SYSTEM
                </button>
            </div>
        </div>
    );
};

export default Sidebar;