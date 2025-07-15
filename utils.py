import pandas as pd
import streamlit as st
import json
from typing import Dict, List, Tuple, Optional

def load_data() -> pd.DataFrame:
    """Load the customer data from CSV file"""
    file_path = r'attached_assets\customer_data.csv'  # Use raw string with backslashes
    return pd.read_csv(file_path)
def check_password(username: str, password: str, users: Dict) -> bool:
    """Verify user credentials - legacy function, use db.validate_user instead"""
    if username in users and users[username] == password:
        return True
    return False

def filter_data(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()

    # Debug print
    if st.session_state.get('show_debug_info', False):
        st.sidebar.write("Applied Filters:", filters)

    if filters.get('age_range'):
        min_age, max_age = filters['age_range']
        filtered_df = filtered_df[
            (filtered_df['age'] >= min_age) & 
            (filtered_df['age'] <= max_age)
        ]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After age filter: {len(filtered_df)} rows")

    if filters.get('income_range'):
        min_income, max_income = filters['income_range']
        filtered_df = filtered_df[
            (filtered_df['income'] >= min_income) & 
            (filtered_df['income'] <= max_income)
        ]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After income filter: {len(filtered_df)} rows")

    if filters.get('gender'):
        filtered_df = filtered_df[filtered_df['gender'].str.lower() == filters['gender'].lower()]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After gender filter: {len(filtered_df)} rows")

    if filters.get('education'):
        filtered_df = filtered_df[filtered_df['education'] == filters['education']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After education filter: {len(filtered_df)} rows")

    if filters.get('region'):
        filtered_df = filtered_df[filtered_df['region'] == filters['region']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After region filter: {len(filtered_df)} rows")

    if filters.get('loyalty_status'):
        filtered_df = filtered_df[filtered_df['loyalty_status'] == filters['loyalty_status']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After loyalty filter: {len(filtered_df)} rows")
            
    if filters.get('purchase_frequency'):
        filtered_df = filtered_df[filtered_df['purchase_frequency'] == filters['purchase_frequency']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After purchase frequency filter: {len(filtered_df)} rows")
            
    if filters.get('product_category'):
        filtered_df = filtered_df[filtered_df['product_category'] == filters['product_category']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After product category filter: {len(filtered_df)} rows")
            
    if filters.get('promotion_usage') is not None:
        filtered_df = filtered_df[filtered_df['promotion_usage'] == filters['promotion_usage']]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After promotion usage filter: {len(filtered_df)} rows")
            
    if filters.get('satisfaction_range'):
        min_score, max_score = filters['satisfaction_range']
        filtered_df = filtered_df[
            (filtered_df['satisfaction_score'] >= min_score) & 
            (filtered_df['satisfaction_score'] <= max_score)
        ]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After satisfaction score filter: {len(filtered_df)} rows")
            
    if filters.get('purchase_amount_range'):
        min_amount, max_amount = filters['purchase_amount_range']
        filtered_df = filtered_df[
            (filtered_df['purchase_amount'] >= min_amount) & 
            (filtered_df['purchase_amount'] <= max_amount)
        ]
        if st.session_state.get('show_debug_info', False):
            st.sidebar.write(f"After purchase amount filter: {len(filtered_df)} rows")

    return filtered_df

def get_unique_values(df: pd.DataFrame) -> Dict:
    """Get unique values for categorical columns"""
    return {
        'gender': sorted(df['gender'].unique().tolist()),
        'education': sorted(df['education'].unique().tolist()),
        'region': sorted(df['region'].unique().tolist()),
        'loyalty_status': sorted(df['loyalty_status'].unique().tolist()),
        'purchase_frequency': sorted(df['purchase_frequency'].unique().tolist()),
        'product_category': sorted(df['product_category'].unique().tolist())
    }
    
def encode_filter_to_json(filters: Dict) -> str:
    """Convert filter dictionary to JSON string for database storage"""
    # Create a copy to avoid modifying the original
    filter_copy = filters.copy()
    
    # Convert tuples to lists for JSON serialization
    for key, value in filter_copy.items():
        if isinstance(value, tuple):
            filter_copy[key] = list(value)
    
    return json.dumps(filter_copy)
    
def decode_filter_from_json(filter_json: str) -> Dict:
    """Convert JSON string back to filter dictionary"""
    filter_dict = json.loads(filter_json)
    
    # Convert lists back to tuples where needed
    for key in ['age_range', 'income_range', 'satisfaction_range', 'purchase_amount_range']:
        if key in filter_dict and isinstance(filter_dict[key], list):
            filter_dict[key] = tuple(filter_dict[key])
    
    return filter_dict