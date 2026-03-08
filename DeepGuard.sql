-- 1. Database Creation
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'DeepGuard_Enterprise')
BEGIN
    CREATE DATABASE DeepGuard_Enterprise;
END
GO

USE DeepGuard_Enterprise;
GO

-- 2. Departments Table (Organisation Hierarchy)
CREATE TABLE Departments (
    dept_id INT PRIMARY KEY IDENTITY(1,1),
    dept_name NVARCHAR(100) UNIQUE NOT NULL,
    dept_head NVARCHAR(100),
    description NVARCHAR(MAX)
);

-- 3. Users Table (Governance-Ready)
CREATE TABLE Users (
    user_id INT PRIMARY KEY IDENTITY(1,1),
    full_name NVARCHAR(100) NOT NULL,
    email NVARCHAR(255) UNIQUE NOT NULL,
    password_hash NVARCHAR(MAX) NOT NULL,
    role NVARCHAR(50) CHECK (role IN ('admin', 'auditor', 'operator')) DEFAULT 'operator',
    dept_id INT FOREIGN KEY REFERENCES Departments(dept_id),
    clearance_level INT CHECK (clearance_level BETWEEN 1 AND 5) DEFAULT 1,
    is_active BIT DEFAULT 1,
    created_at DATETIME DEFAULT GETDATE()
);

-- 4. Governance Policies (The Rules of the System)
CREATE TABLE GovernancePolicies (
    policy_id INT PRIMARY KEY IDENTITY(1,1),
    policy_title NVARCHAR(200),
    min_confidence_threshold FLOAT DEFAULT 80.0, -- Isse kam par flag hoga
    alert_action NVARCHAR(100) DEFAULT 'NOTIFY_ADMIN',
    is_mandatory BIT DEFAULT 1,
    last_modified_by INT FOREIGN KEY REFERENCES Users(user_id)
);

-- 5. Forensic Analysis History
CREATE TABLE AnalysisHistory (
    analysis_id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT FOREIGN KEY REFERENCES Users(user_id),
    filename NVARCHAR(255) NOT NULL,
    verdict NVARCHAR(50), -- REAL / FAKE
    confidence_score FLOAT,
    -- AI Branch Scores
    spatial_score FLOAT,
    physics_score FLOAT,
    forensics_score FLOAT,
    audio_score FLOAT,
    -- Processing Data
    processing_time_sec FLOAT,
    client_ip NVARCHAR(50),
    timestamp DATETIME DEFAULT GETDATE()
);

-- 6. Video Metadata (Technical Forensics)
CREATE TABLE VideoMetadata (
    meta_id INT PRIMARY KEY IDENTITY(1,1),
    analysis_id INT FOREIGN KEY REFERENCES AnalysisHistory(analysis_id) ON DELETE CASCADE,
    file_size_mb FLOAT,
    resolution NVARCHAR(50),
    codec NVARCHAR(50),
    fps INT,
    duration_sec FLOAT
);

-- 7. Verification Workflow (Governance Verification)
-- Jab AI detect kare, toh Auditor ka sign-off zaroori hai
CREATE TABLE ResultVerifications (
    verify_id INT PRIMARY KEY IDENTITY(1,1),
    analysis_id INT FOREIGN KEY REFERENCES AnalysisHistory(analysis_id),
    auditor_id INT FOREIGN KEY REFERENCES Users(user_id),
    final_verdict NVARCHAR(50), -- 'CONFIRMED', 'OVERTURNED'
    verification_status NVARCHAR(50) DEFAULT 'PENDING', -- PENDING, COMPLETED
    comments NVARCHAR(MAX),
    verified_at DATETIME
);

-- 8. Audit Logs (Compliance & Accountability)
CREATE TABLE AuditLogs (
    log_id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT FOREIGN KEY REFERENCES Users(user_id),
    action_type NVARCHAR(100), -- 'LOGIN', 'UPLOAD', 'POLICY_CHANGE', 'VERIFY'
    description NVARCHAR(MAX),
    ip_address NVARCHAR(50),
    created_at DATETIME DEFAULT GETDATE()
);

-- 9. Initial Governance Data (Seeds)
-- Pehle department aur ek admin banana zaroori hai
INSERT INTO Departments (dept_name, description) VALUES ('Cyber Forensics Lab', 'Main analysis hub');

-- Admin Profile (Password: admin123 - Baad mein hash kar lena)
INSERT INTO Users (full_name, email, password_hash, role, dept_id, clearance_level)
VALUES ('Super Admin', 'admin@deepguard.com', 'admin123', 'admin', 1, 5);

INSERT INTO GovernancePolicies (policy_title, min_confidence_threshold, alert_action)
VALUES ('Standard Deepfake Detection Policy', 85.0, 'MANDATORY_VERIFICATION');
GO

select*
from dbo.AnalysisHistory