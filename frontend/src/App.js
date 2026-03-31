import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import UserLogin from "./pages/UserLogin";
import AdminLogin from "./pages/AdminLogin";
import Register from "./pages/Register"; // Naya Import
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import UserManagement from "./pages/UserManagement";
import AssignMember from "./pages/AssignMember";

function App() {
  return (
    <Router>
      <Routes>
        {/* Gateway Routes */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/register" element={<Register />} /> {/* Is line ki wajah se khali page ara tha */}
        <Route path="/login-user" element={<UserLogin />} />
        <Route path="/login-admin" element={<AdminLogin />} />

        {/* Protected Routes (Inka Layout aap ne pehle set kiya hua hai) */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/user-management" element={<UserManagement />} />
        <Route path="/assign-member" element={<AssignMember />} />
      </Routes>
    </Router>
  );
}

export default App;