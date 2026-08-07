-- 1. HITL Validation Queue (For Definitions Pipeline & Cross-Encoder Evaluation)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'HITL_Validation_Queue')
BEGIN
    CREATE TABLE HITL_Validation_Queue (
        QueueID INT IDENTITY(1,1) PRIMARY KEY,
        skill_name NVARCHAR(150) NOT NULL,
        category NVARCHAR(100),
        
        -- Wikipedia Source Payload & Score
        wiki_title NVARCHAR(200),
        wiki_summary NVARCHAR(MAX),
        wiki_score DECIMAL(5, 4),

        -- GitHub Source Payload & Score
        github_title NVARCHAR(200),
        github_summary NVARCHAR(MAX),
        github_score DECIMAL(5, 4),

        -- PyPI Source Payload & Score
        pypi_title NVARCHAR(200),
        pypi_summary NVARCHAR(MAX),
        pypi_score DECIMAL(5, 4),

        -- O*NET Taxonomy Metadata
        onet_code NVARCHAR(50),
        onet_title NVARCHAR(150),
        is_hot_tech BIT DEFAULT 0,
        
        -- Approval Workflow (1 = Auto-Approved [all scores >= 0.90], 0 = Flagged for HITL Review)
        is_approved BIT DEFAULT 0,
        created_at DATETIME DEFAULT GETDATE()
    );
END;

-- 2. Master Skills Inventory
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Skills')
BEGIN
    CREATE TABLE Skills (
        SkillID INT IDENTITY(1,1) PRIMARY KEY,
        SkillName NVARCHAR(150) NOT NULL UNIQUE,
        Category NVARCHAR(100),
        Description NVARCHAR(MAX),
        IsApproved BIT DEFAULT 0,
        CreatedAt DATETIME DEFAULT GETDATE()
    );
END;

-- 3. Users Table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Users')
BEGIN
    CREATE TABLE Users (
        UserID INT IDENTITY(1,1) PRIMARY KEY,
        Username NVARCHAR(50) NOT NULL UNIQUE,
        Email NVARCHAR(100) NOT NULL UNIQUE,
        CreatedAt DATETIME DEFAULT GETDATE()
    );
END;

-- 4. User Skill Scores / Analytics
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'UserScores')
BEGIN
    CREATE TABLE UserScores (
        ScoreID INT IDENTITY(1,1) PRIMARY KEY,
        UserID INT FOREIGN KEY REFERENCES Users(UserID) ON DELETE CASCADE,
        SkillID INT FOREIGN KEY REFERENCES Skills(SkillID) ON DELETE CASCADE,
        ScoreValue DECIMAL(5, 2) NOT NULL,
        EvaluatedAt DATETIME DEFAULT GETDATE()
    );
END;

-- 5. O*NET Taxonomy Reference
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'OnetTaxonomy')
BEGIN
    CREATE TABLE OnetTaxonomy (
        OnetCode NVARCHAR(50) PRIMARY KEY,
        OnetTitle NVARCHAR(150) NOT NULL,
        JobFamily NVARCHAR(100),
        LastUpdated DATETIME DEFAULT GETDATE()
    );
END;