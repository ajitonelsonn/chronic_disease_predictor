import mysql.connector
import streamlit as st
import logging
from typing import Dict, List
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection():
    """Create a database connection using Streamlit secrets"""
    try:
        connection = mysql.connector.connect(
            host=st.secrets["database"]["db_host"],
            user=st.secrets["database"]["db_username"],
            password=st.secrets["database"]["db_password"],
            database=st.secrets["database"]["db_name"],
            port=int(st.secrets["database"]["db_port"])
        )
        return connection
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        st.error(f"Failed to connect to database: {err}")
        raise

def convert_numpy_types(value):
    """Convert numpy types to Python native types"""
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def save_prediction_to_database(patient_data: Dict, prediction_results: Dict, recommendations: str) -> bool:
    """
    Save prediction results to database
    
    Args:
        patient_data: Dictionary containing patient information
        prediction_results: Dictionary containing prediction results
        recommendations: String containing LLM recommendations
    
    Returns:
        bool: True if save successful, False otherwise
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Insert patient data
        patient_query = """
        INSERT INTO patients (age, gender, race, ethnicity)
        VALUES (%s, %s, %s, %s)
        """
        patient_values = (
            patient_data['age'],
            patient_data['gender'],
            patient_data['race'],
            patient_data['ethnicity']
        )
        cursor.execute(patient_query, patient_values)
        patient_id = cursor.lastrowid

        # 2. Insert patient conditions
        if patient_data['conditions']:
            # Get condition IDs
            condition_query = "SELECT condition_id, condition_name FROM conditions WHERE condition_name IN ({})".format(
                ','.join(['%s'] * len(patient_data['conditions']))
            )
            cursor.execute(condition_query, tuple(patient_data['conditions']))
            condition_ids = cursor.fetchall()

            # Insert into patient_conditions
            for condition_id, _ in condition_ids:
                cursor.execute("""
                    INSERT INTO patient_conditions (patient_id, condition_id)
                    VALUES (%s, %s)
                """, (patient_id, condition_id))

        # 3. Insert prediction
        prediction_query = """
        INSERT INTO predictions (patient_id, primary_condition, confidence, risk_level)
        VALUES (%s, %s, %s, %s)
        """
        risk_level = "High" if len(patient_data['conditions']) >= 3 else \
                    "Medium" if len(patient_data['conditions']) >= 1 else "Low"
        
        # Convert confidence to Python float
        confidence = convert_numpy_types(prediction_results['confidence'])
        
        prediction_values = (
            patient_id,
            prediction_results['primary_condition'],
            confidence,
            risk_level
        )
        cursor.execute(prediction_query, prediction_values)
        prediction_id = cursor.lastrowid

        # 4. Insert prediction details (top 3)
        detail_query = """
        INSERT INTO prediction_details (prediction_id, condition_name, probability, prediction_rank)
        VALUES (%s, %s, %s, %s)
        """
        for rank, (condition, prob) in enumerate(prediction_results['top_3_conditions'], 1):
            # Convert probability to Python float
            probability = convert_numpy_types(prob)
            cursor.execute(detail_query, (prediction_id, condition, probability, rank))

        # 5. Insert recommendations
        if recommendations:
            recommend_query = """
            INSERT INTO recommendations (prediction_id, recommendation_text)
            VALUES (%s, %s)
            """
            cursor.execute(recommend_query, (prediction_id, recommendations))

        conn.commit()
        logger.info("Successfully saved prediction to database")
        return True

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logger.error(f"General error: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_saved_predictions() -> List[Dict]:
    """
    Retrieve saved predictions from database
    
    Returns:
        List of dictionaries containing prediction information
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT
            p.patient_id,
            p.age,
            p.gender,
            p.race,
            p.ethnicity,
            pred.prediction_id,
            pred.primary_condition,
            pred.confidence,
            pred.risk_level,
            pred.created_at,
            GROUP_CONCAT(c.condition_name) AS conditions
        FROM
            patients p
        JOIN
            predictions pred ON p.patient_id = pred.patient_id
        LEFT JOIN
            patient_conditions pc ON p.patient_id = pc.patient_id
        LEFT JOIN
            conditions c ON pc.condition_id = c.condition_id
        GROUP BY
            p.patient_id, p.age, p.gender, p.race, p.ethnicity,
            pred.prediction_id, pred.primary_condition, pred.confidence, pred.risk_level, pred.created_at
        ORDER BY
            pred.created_at DESC
        """
        cursor.execute(query)
        return cursor.fetchall()

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def test_connection() -> bool:
    """Test the database connection"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        conn.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False