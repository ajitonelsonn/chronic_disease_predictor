import pandas as pd

def load_condition_mapping():
    """Load the condition code to description mapping"""
    try:
        mapping_df = pd.read_csv('data/PRIMARY_CHRONIC_CONDITION_ROLLUP_DESC.csv')
        return dict(zip(mapping_df['PCC_CODE'], mapping_df['PCC_ROLLUP']))
    except Exception:
        return {}

def get_condition_name(code, condition_map):
    """Convert condition code to actual condition name"""
    return condition_map.get(float(code), f"Unknown ({code})")

def get_feature_names():
    """Get expected feature names in correct order"""
    return [
        'MEM_GENDER_ENCODED',
        'MEM_RACE_ENCODED',
        'MEM_ETHNICITY_ENCODED',
        'PAYER_LOB_ENCODED',
        'SERVICE_SETTING_ENCODED',
        'DIAGNOSTIC_CONDITION_CATEGORY_DESC_ENCODED',
        'HAS_HYPERTENSION',
        'HAS_DIABETES',
        'HAS_RENAL_FAILURE',
        'HAS_OTHER_CHRONIC',
        'HAS_CANCER',
        'HAS_MENTAL_HEALTH',
        'HAS_HEART_FAILURE',
        'HAS_ASTHMA',
        'HAS_MUSCULOSKELETAL',
        'HAS_NEUROLOGIC',
        'HAS_LIVER_DISEASE',
        'HAS_DEMENTIA',
        'HAS_CAD',
        'HAS_HYPERTENSION_AND_DIABETES',
        'HAS_HYPERTENSION_AND_HEART_FAILURE',
        'HAS_DIABETES_AND_RENAL_FAILURE',
        'HAS_HEART_FAILURE_AND_RENAL_FAILURE',
        'HAS_MENTAL_HEALTH_AND_NEUROLOGIC',
        'HAS_LIVER_DISEASE_AND_RENAL_FAILURE',
        'MEM_AGE_NUMERIC',
        'WEIGHTED_RISK_SCORE'
    ]