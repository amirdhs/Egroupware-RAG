-- EGroupware RAG Database Initialization Script
-- This script creates the necessary tables for vector storage

CREATE DATABASE IF NOT EXISTS rag_vectors;
USE rag_vectors;

-- Single documents table with embeddings stored directly
-- This matches what the application code expects
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    doc_id VARCHAR(255) NOT NULL,
    app_name VARCHAR(50) NOT NULL,
    content LONGTEXT NOT NULL,
    embedding LONGBLOB NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_app_name (app_name),
    INDEX idx_user_app (user_id, app_name),
    INDEX idx_created_at (created_at),
    UNIQUE KEY unique_user_app_doc (user_id, app_name, doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Grant permissions to the rag_user
GRANT ALL PRIVILEGES ON rag_vectors.* TO 'rag_user'@'%';
FLUSH PRIVILEGES;
