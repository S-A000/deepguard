import React from 'react';

const MediaAnalysisHub = ({ file, setFile, previewUrl, setPreviewUrl, setResult }) => {
  return (
    <div className="flex flex-col gap-4">
      {/* Title */}
      <div className="flex justify-between items-center px-1">
        <h2 className="text-lg font-semibold text-white">Media Analysis Hub</h2>
        <button className="text-gray-500 hover:text-white">...</button>
      </div>
      
      {/* Upload Center */}
      <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-6 flex flex-col items-center justify-center text-center relative shadow-lg">
        <div className="absolute inset-x-0 -top-px h-px w-1/2 mx-auto bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50"></div>
        <div className="absolute top-0 left-0 w-full h-24 bg-gradient-to-b from-cyan-500/10 to-transparent pointer-events-none rounded-t-2xl"></div>
        <div className="border border-dashed border-gray-600 rounded-xl p-8 w-full flex flex-col items-center justify-center hover:border-cyan-500 transition-colors z-10 relative">
          <svg className="w-8 h-8 text-cyan-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          <h3 className="text-white font-medium mb-1 z-10">Upload Center</h3>
          <p className="text-gray-400 text-xs z-10">Drag & Drop Video or Image</p>
          <p className="text-gray-500 text-[10px] mb-4 z-10">(Max 5GB)</p>
          <label className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white px-6 py-2 rounded-lg text-sm font-medium cursor-pointer transition-all shadow-[0_0_15px_rgba(6,182,212,0.4)] z-10">
            Choose File
            <input 
              type="file" 
              className="hidden" 
              accept="video/*,image/*"
              onChange={(e) => {
                if (e.target.files[0]) {
                  setFile(e.target.files[0]);
                  setPreviewUrl(URL.createObjectURL(e.target.files[0]));
                  setResult(null);
                }
              }} 
            />
          </label>
        </div>
      </div>

      {/* Recently Uploaded Video */}
      <div className="bg-[#11131a] rounded-2xl border border-[#1f2937] p-5 relative overflow-hidden shadow-lg mt-1">
        <div className="absolute inset-x-0 -top-px h-px w-1/2 mx-auto bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-40"></div>
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-cyan-500/5 to-transparent pointer-events-none"></div>
        
        <div className="flex justify-between items-center mb-4 relative z-10">
          <h3 className="text-white font-medium">Recently Uploaded Video</h3>
          <button className="text-gray-500 hover:text-white">...</button>
        </div>

        <div className="relative rounded-xl overflow-hidden bg-black aspect-video group flex items-center justify-center border border-gray-800">
            {previewUrl ? (
                file?.type?.includes('image') ? (
                    <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                ) : (
                    <video src={previewUrl} className="w-full h-full object-cover" controls={false} />
                )
            ) : (
                <div className="w-full h-full bg-[#0a0f1c] flex items-center justify-center">
                    <span className="text-gray-700 text-xs">No media</span>
                </div>
            )}
            
            {/* Play Button Overlay (mock) */}
            {(previewUrl && !file?.type?.includes('image')) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30 group-hover:bg-black/10 transition-all pointer-events-none">
                    <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center">
                        <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-lg">
                            <svg className="w-5 h-5 text-black ml-1" fill="currentColor" viewBox="0 0 20 20"><path d="M4 4l12 6-12 6z" /></svg>
                        </div>
                    </div>
                </div>
            )}
        </div>

        <div className="mt-4 relative z-10">
          <p className="text-white text-sm font-medium truncate">{file ? file.name : "Deepfake_Sample_01.mp4"}</p>
          <p className="text-gray-500 text-xs mb-3">Thumbnail</p>
          
          {/* Custom Player Controls (Mock) */}
          <div className="flex items-center gap-3">
            <button className="text-white hover:text-cyan-400 focus:outline-none">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M4 4l12 6-12 6z" /></svg>
            </button>
            <div className="flex-1 h-1 bg-gray-700 rounded-full relative overflow-hidden">
               <div className="absolute left-0 top-0 h-full w-[45%] bg-gradient-to-r from-purple-500 to-cyan-400 rounded-full"></div>
            </div>
            <span className="text-gray-400 text-[10px]">00:32 / 01:15</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MediaAnalysisHub;
