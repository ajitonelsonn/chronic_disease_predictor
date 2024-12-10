-- database_setup.sql

-- Create the database
CREATE DATABASE IF NOT EXISTS chronic_disease_predictor;
USE chronic_disease_predictor;

-- Create patients table to store demographic information
CREATE TABLE patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    race VARCHAR(50) NOT NULL,xa
    ethnicity VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create conditions table to store available conditions
CREATE TABLE conditions (
    condition_id INT AUTO_INCREMENT PRIMARY KEY,
    condition_name VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create patient_conditions table for many-to-many relationship
CREATE TABLE patient_conditions (
    patient_condition_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    condition_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (condition_id) REFERENCES conditions(condition_id)
);

-- Create predictions table to store model predictions
CREATE TABLE predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    primary_condition VARCHAR(200) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    risk_level ENUM('Low', 'Medium', 'High') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Create prediction_details table for storing top 3 predictions
CREATE TABLE prediction_details (
    detail_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    condition_name VARCHAR(200) NOT NULL,
    probability DECIMAL(5,2) NOT NULL,
    prediction_rank INT NOT NULL,  -- Changed from 'rank' as it's a reserved word
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- Create recommendations table to store LLM-generated recommendations
CREATE TABLE recommendations (
    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    recommendation_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

-- Insert initial conditions
INSERT INTO conditions (condition_name) VALUES 
    ('Hypertension'),
    ('Diabetes'),
    ('Cancer'),
    ('Renal Failure'),
    ('Depression'),
    ('Heart Failure'),
    ('Asthma'),
    ('CAD'),
    ('Musculoskeletal'),
    ('Neurologic'),
    ('Liver Disease'),
    ('Dementia');