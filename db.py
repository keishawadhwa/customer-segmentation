import os
import pandas as pd
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

# Flag to track database availability
db_available = False

# Initialize session state variables if they don't exist
if not hasattr(st.session_state, 'users'):
    st.session_state.users = {'admin': 'admin'}

try:
    # Get database connection string from environment variable
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    if DATABASE_URL:
        # Create engine with a timeout
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 3})
        Base = declarative_base()
        
        # Try a quick connection test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_available = True
    else:
        print("DATABASE_URL not set in environment variables")
except Exception as e:
    print(f"Database connection error: {str(e)}")
    db_available = False

# Only define models if database is available
if db_available:
    # Define User model for authentication
    class User(Base):
        __tablename__ = 'users'
        
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True, nullable=False)
        password = Column(String(100), nullable=False)
        
        def __repr__(self):
            return f"<User(username='{self.username}')>"

    # Define SavedFilter model for storing user filters
    class SavedFilter(Base):
        __tablename__ = 'saved_filters'
        
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        name = Column(String(100), nullable=False)
        filter_json = Column(Text, nullable=False)
        
        def __repr__(self):
            return f"<SavedFilter(name='{self.name}')>"

    # Create tables
    try:
        Base.metadata.create_all(engine)
        # Create session factory
        Session = sessionmaker(bind=engine)
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        db_available = False

# Dictionary to store session-based user data when DB is unavailable
session_users = {}
session_filters = {}

def is_db_available():
    """Check if database is available"""
    return db_available

def get_db_session():
    """Get a database session"""
    if not db_available:
        raise Exception("Database not available")
    return Session()

def add_user(username, password):
    """Add a new user to the database or session state"""
    if db_available:
        session = get_db_session()
        try:
            # Check if user already exists
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                session.close()
                return False, "Username already exists"
            
            # Create new user
            new_user = User(username=username, password=password)
            session.add(new_user)
            session.commit()
            return True, "User created successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error creating user: {str(e)}"
        finally:
            session.close()
    else:
        # Fallback to session state
        if username in st.session_state.users:
            return False, "Username already exists"
        
        # Add to session state
        st.session_state.users[username] = password
        return True, "User created successfully (stored in session)"

def validate_user(username, password):
    """Validate user credentials"""
    if db_available:
        session = get_db_session()
        try:
            user = session.query(User).filter_by(username=username).first()
            if user and user.password == password:
                return True, user.id
            return False, None
        except Exception as e:
            return False, None
        finally:
            session.close()
    else:
        # Fallback to session state
        if username in st.session_state.users and st.session_state.users[username] == password:
            return True, username  # Use username as ID for session-based auth
        return False, None

def save_filter(user_id, filter_name, filter_data):
    """Save a filter for a user"""
    if db_available:
        session = get_db_session()
        try:
            # Check if filter name already exists for this user
            existing_filter = session.query(SavedFilter).filter_by(
                user_id=user_id, name=filter_name).first()
            
            if existing_filter:
                # Update existing filter
                existing_filter.filter_json = filter_data
                session.commit()
                return True, "Filter updated successfully"
            else:
                # Create new filter
                new_filter = SavedFilter(
                    user_id=user_id,
                    name=filter_name,
                    filter_json=filter_data
                )
                session.add(new_filter)
                session.commit()
                return True, "Filter saved successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error saving filter: {str(e)}"
        finally:
            session.close()
    else:
        # Fallback to session state
        user_key = str(user_id)  # Convert ID to string for dictionary key
        
        if user_key not in session_filters:
            session_filters[user_key] = {}
            
        session_filters[user_key][filter_name] = filter_data
        return True, "Filter saved successfully (stored in session)"

def get_user_filters(user_id):
    """Get all saved filters for a user"""
    if db_available:
        session = get_db_session()
        try:
            filters = session.query(SavedFilter).filter_by(user_id=user_id).all()
            return [(f.name, f.filter_json) for f in filters]
        except Exception as e:
            return []
        finally:
            session.close()
    else:
        # Fallback to session state
        user_key = str(user_id)  # Convert ID to string for dictionary key
        
        if user_key in session_filters:
            return [(name, data) for name, data in session_filters[user_key].items()]
        return []

def delete_filter(user_id, filter_name):
    """Delete a saved filter"""
    if db_available:
        session = get_db_session()
        try:
            filter_to_delete = session.query(SavedFilter).filter_by(
                user_id=user_id, name=filter_name).first()
            
            if filter_to_delete:
                session.delete(filter_to_delete)
                session.commit()
                return True, "Filter deleted successfully"
            else:
                return False, "Filter not found"
        except Exception as e:
            session.rollback()
            return False, f"Error deleting filter: {str(e)}"
        finally:
            session.close()
    else:
        # Fallback to session state
        user_key = str(user_id)  # Convert ID to string for dictionary key
        
        if user_key in session_filters and filter_name in session_filters[user_key]:
            del session_filters[user_key][filter_name]
            return True, "Filter deleted successfully (from session)"
        return False, "Filter not found"

def migrate_existing_users():
    """Migrate existing users from session state to database"""
    if not db_available or 'users' not in st.session_state:
        return
        
    session = get_db_session()
    
    try:
        for username, password in st.session_state.users.items():
            # Check if user already exists in DB
            existing_user = session.query(User).filter_by(username=username).first()
            if not existing_user:
                # Add user to DB
                new_user = User(username=username, password=password)
                session.add(new_user)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error migrating users: {str(e)}")
    finally:
        session.close()