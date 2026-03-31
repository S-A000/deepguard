// Iska kaam hai check karna ke koi login hai ya nahi
export const isAuthenticated = () => {
    const role = localStorage.getItem('userRole');
    return role !== null;
};

// Logout ka kaam
export const logout = () => {
    localStorage.clear();
    window.location.href = "/";
};

// User ka info nikalna
export const getUserInfo = () => {
    return {
        role: localStorage.getItem('userRole'),
        email: localStorage.getItem('userEmail')
    };
};