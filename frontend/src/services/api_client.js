// frontend/src/services/api_client.js

export const uploadAndAnalyzeVideo = async (videoFile) => {
    const formData = new FormData();
    formData.append("file", videoFile);

    try {
        // Yeh tumhare FastAPI backend ka address hai
        const response = await fetch("http://127.0.0.1:8000/api/analyze", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Server response failed. Check if Backend is running.");
        }

        const data = await response.json();
        return data; 
    } catch (error) {
        console.error("API Error:", error);
        return { status: "error", message: error.message };
    }
};