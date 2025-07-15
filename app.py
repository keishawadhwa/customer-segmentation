import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from utils import load_data, check_password, filter_data, get_unique_values, encode_filter_to_json, decode_filter_from_json
import db
import clustering
import base64
from io import BytesIO

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'users' not in st.session_state:
    st.session_state.users = {'admin': 'admin'}  # Default user (legacy)
if 'data' not in st.session_state:
    st.session_state.data = load_data()
if 'current_filter' not in st.session_state:
    st.session_state.current_filter = {}
if 'show_debug_info' not in st.session_state:
    st.session_state.show_debug_info = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'cluster_stats' not in st.session_state:
    st.session_state.cluster_stats = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3

# Try to migrate legacy users to database
try:
    db.migrate_existing_users()
except Exception as e:
    if st.session_state.show_debug_info:
        st.error(f"Error migrating users: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    st.title("Customer Data Filter & Segmentation App")
    
    # Show toggle for debug info in a small container
    with st.expander("Debug Options", expanded=False):
        st.session_state.show_debug_info = st.checkbox(
            "Show Debug Information", 
            value=st.session_state.show_debug_info
        )

    if not st.session_state.authenticated:
        auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])

        with auth_tab1:
            st.header("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", key="login_button"):
                # Try database authentication first
                try:
                    success, user_id = db.validate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        st.rerun()
                except Exception as e:
                    if st.session_state.show_debug_info:
                        st.warning(f"Database login error: {str(e)}")
                        
                # Fall back to legacy authentication
                if check_password(username, password, st.session_state.users):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with auth_tab2:
            st.header("Sign Up")
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

            if st.button("Sign Up", key="signup_button"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Try to add user to database
                    try:
                        success, message = db.add_user(new_username, new_password)
                        if success:
                            st.success(f"{message}! Please login.")
                        else:
                            st.error(message)
                    except Exception as e:
                        # Fall back to session state
                        if new_username in st.session_state.users:
                            st.error("Username already exists")
                        else:
                            st.session_state.users[new_username] = new_password
                            st.success("Account created successfully! Please login.")
                            if st.session_state.show_debug_info:
                                st.info("User saved to session (database unavailable)")

    else:
        # Create tabs for main functionality
        tab1, tab2, tab3, tab4 = st.tabs([
            "Filter Customers", 
            "Customer Segmentation", 
            "Segment Analysis", 
            "Saved Filters"
        ])
        
        # Get unique values for filters
        unique_values = get_unique_values(st.session_state.data)
        
        # Get filtered data for all tabs
        filtered_df = filter_data(st.session_state.data, st.session_state.current_filter)
        
        # First tab - Main filtering interface with side-by-side layout
        with tab1:
            # Create a two-column layout: filters on the left, results on the right
            filter_col, results_col = st.columns([1, 3])
            
            with filter_col:
                st.header("Filter Options")
                
                # Basic filters
                st.subheader("Demographics")
                
                # Age filter
                age_min = int(st.session_state.data['age'].min())
                age_max = int(st.session_state.data['age'].max())
                age_range = st.slider(
                    "Age Range",
                    min_value=age_min,
                    max_value=age_max,
                    value=st.session_state.current_filter.get('age_range', (age_min, age_max)),
                    key='main_age_filter'
                )
                st.session_state.current_filter['age_range'] = age_range

                # Gender filter
                gender = st.selectbox(
                    "Gender", 
                    ["All"] + unique_values['gender'],
                    index=0 if "gender" not in st.session_state.current_filter else 
                         (unique_values['gender'].index(st.session_state.current_filter['gender']) + 1),
                    key='main_gender_filter'
                )
                if gender != "All":
                    st.session_state.current_filter['gender'] = gender
                elif 'gender' in st.session_state.current_filter:
                    del st.session_state.current_filter['gender']
                
                # Education filter
                education = st.selectbox(
                    "Education", 
                    ["All"] + unique_values['education'],
                    index=0 if "education" not in st.session_state.current_filter else 
                         (unique_values['education'].index(st.session_state.current_filter['education']) + 1),
                    key='main_education_filter'
                )
                if education != "All":
                    st.session_state.current_filter['education'] = education
                elif 'education' in st.session_state.current_filter:
                    del st.session_state.current_filter['education']
                
                # Region filter
                region = st.selectbox(
                    "Region", 
                    ["All"] + unique_values['region'],
                    index=0 if "region" not in st.session_state.current_filter else 
                         (unique_values['region'].index(st.session_state.current_filter['region']) + 1),
                    key='main_region_filter'
                )
                if region != "All":
                    st.session_state.current_filter['region'] = region
                elif 'region' in st.session_state.current_filter:
                    del st.session_state.current_filter['region']
                
                st.subheader("Financial")
                # Income filter
                income_min = int(st.session_state.data['income'].min())
                income_max = int(st.session_state.data['income'].max())
                income_range = st.slider(
                    "Income Range ($)",
                    min_value=income_min,
                    max_value=income_max,
                    value=st.session_state.current_filter.get('income_range', (income_min, income_max)),
                    key='main_income_filter'
                )
                st.session_state.current_filter['income_range'] = income_range
                
                # Purchase Amount filter
                amount_min = int(st.session_state.data['purchase_amount'].min())
                amount_max = int(st.session_state.data['purchase_amount'].max())
                purchase_amount_range = st.slider(
                    "Purchase Amount ($)",
                    min_value=amount_min,
                    max_value=amount_max,
                    value=st.session_state.current_filter.get('purchase_amount_range', (amount_min, amount_max)),
                    key='main_purchase_filter'
                )
                st.session_state.current_filter['purchase_amount_range'] = purchase_amount_range
                
                st.subheader("Purchase Behavior")
                # Loyalty Status filter
                loyalty = st.selectbox(
                    "Loyalty Status", 
                    ["All"] + unique_values['loyalty_status'],
                    index=0 if "loyalty_status" not in st.session_state.current_filter else 
                         (unique_values['loyalty_status'].index(st.session_state.current_filter['loyalty_status']) + 1),
                    key='main_loyalty_filter'
                )
                if loyalty != "All":
                    st.session_state.current_filter['loyalty_status'] = loyalty
                elif 'loyalty_status' in st.session_state.current_filter:
                    del st.session_state.current_filter['loyalty_status']
                
                # Purchase Frequency filter
                frequency = st.selectbox(
                    "Purchase Frequency", 
                    ["All"] + unique_values['purchase_frequency'],
                    index=0 if "purchase_frequency" not in st.session_state.current_filter else 
                         (unique_values['purchase_frequency'].index(st.session_state.current_filter['purchase_frequency']) + 1),
                    key='main_frequency_filter'
                )
                if frequency != "All":
                    st.session_state.current_filter['purchase_frequency'] = frequency
                elif 'purchase_frequency' in st.session_state.current_filter:
                    del st.session_state.current_filter['purchase_frequency']
                
                # Product Category filter
                category = st.selectbox(
                    "Product Category", 
                    ["All"] + unique_values['product_category'],
                    index=0 if "product_category" not in st.session_state.current_filter else 
                         (unique_values['product_category'].index(st.session_state.current_filter['product_category']) + 1),
                    key='main_category_filter'
                )
                if category != "All":
                    st.session_state.current_filter['product_category'] = category
                elif 'product_category' in st.session_state.current_filter:
                    del st.session_state.current_filter['product_category']
                
                # Save and reset filters
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reset Filters", key="main_reset_filters"):
                        st.session_state.current_filter = {}
                        st.rerun()
                
                with col2:
                    if st.session_state.user_id and st.button("Save Filter", key="main_save_filter"):
                        st.session_state.active_tab = 3  # Switch to saved filters tab
                        st.rerun()
            
            # Right column for results and visualizations
            with results_col:
                # Display filtered results
                filtered_df = filter_data(st.session_state.data, st.session_state.current_filter)
                st.header("Results")
                st.write(f"Found {len(filtered_df)} matching customers")

                if not filtered_df.empty:
                    # Tabs for different views of the data
                    results_tab1, results_tab2, results_tab3 = st.tabs(["Table View", "Charts", "Quick Segmentation"])
                    
                    with results_tab1:
                        # Column selector
                        cols_to_display = st.multiselect(
                            "Select columns to display",
                            filtered_df.columns.tolist(),
                            default=['id', 'age', 'gender', 'income', 'education', 'region', 'loyalty_status']
                        )

                        if cols_to_display:
                            st.dataframe(filtered_df[cols_to_display], use_container_width=True)
                        else:
                            st.warning("Please select at least one column to display")
                    
                    with results_tab2:
                        # Add visualization options
                        chart_type = st.selectbox(
                            "Select Chart Type",
                            ["Distribution", "Correlation", "Count by Category"]
                        )
                        
                        if chart_type == "Distribution":
                            # Let user select a numeric column to visualize distribution
                            numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            dist_col = st.selectbox("Select Column", numeric_cols)
                            
                            # Plot histogram
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.hist(filtered_df[dist_col], bins=30, alpha=0.7)
                            ax.set_xlabel(dist_col)
                            ax.set_ylabel("Frequency")
                            ax.set_title(f"Distribution of {dist_col}")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # Display basic statistics
                            st.subheader(f"Statistics for {dist_col}")
                            stats_df = pd.DataFrame({
                                'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                                'Value': [
                                    f"{filtered_df[dist_col].mean():.2f}",
                                    f"{filtered_df[dist_col].median():.2f}",
                                    f"{filtered_df[dist_col].min():.2f}",
                                    f"{filtered_df[dist_col].max():.2f}",
                                    f"{filtered_df[dist_col].std():.2f}"
                                ]
                            })
                            st.table(stats_df)
                            
                        elif chart_type == "Correlation":
                            # Let user select two numeric columns to visualize correlation
                            numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("X-axis", numeric_cols, index=0)
                            with col2:
                                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                            
                            # Plot scatter plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.5)
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.set_title(f"{y_col} vs {x_col}")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # Display correlation coefficient
                            corr = filtered_df[[x_col, y_col]].corr().iloc[0, 1]
                            st.info(f"Correlation coefficient: {corr:.3f}")
                            
                        elif chart_type == "Count by Category":
                            # Let user select a categorical column to show counts
                            cat_cols = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category']
                            valid_cat_cols = [col for col in cat_cols if col in filtered_df.columns]
                            cat_col = st.selectbox("Select Category", valid_cat_cols)
                            
                            # Calculate and plot counts
                            value_counts = filtered_df[cat_col].value_counts()
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.bar(value_counts.index, value_counts.values)
                            ax.set_xlabel(cat_col)
                            ax.set_ylabel("Count")
                            ax.set_title(f"Count by {cat_col}")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Display as a table as well
                            st.subheader(f"Count by {cat_col}")
                            count_df = pd.DataFrame({
                                cat_col: value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / value_counts.sum() * 100).round(1)
                            })
                            st.table(count_df)
                    
                    with results_tab3:
                        st.subheader("Quick Segmentation")
                        st.write("Create segments from your filtered data using K-means clustering.")
                        
                        # Quick clustring controls
                        col1, col2 = st.columns(2)
                        with col1:
                            quick_segments = st.slider("Number of segments", 2, 5, 3)
                        with col2:
                            if st.button("Run Quick Segmentation"):
                                # Use age, income, purchase_amount, and satisfaction_score by default
                                default_features = [col for col in ['age', 'income', 'purchase_amount', 'satisfaction_score'] 
                                                  if col in filtered_df.columns]
                                
                                if len(default_features) >= 2:
                                    with st.spinner("Generating segments..."):
                                        try:
                                            clustered_df, cluster_stats = clustering.apply_kmeans_clustering(
                                                filtered_df, 
                                                n_clusters=quick_segments,
                                                numeric_columns=default_features
                                            )
                                            
                                            if clustered_df is not None:
                                                st.session_state.clustered_data = clustered_df
                                                st.session_state.cluster_stats = cluster_stats
                                                st.success(f"Successfully created {quick_segments} segments!")
                                                
                                                # Display segment sizes
                                                segment_sizes = {}
                                                if isinstance(cluster_stats, dict) and 'cluster_sizes' in cluster_stats:
                                                    segment_sizes = cluster_stats['cluster_sizes']
                                                
                                                # Create segment labels and values for the chart
                                                segment_labels = [f"Segment {i+1}" for i in range(len(segment_sizes))]
                                                segment_values = list(segment_sizes.values()) if segment_sizes else []
                                                
                                                # Plot segment sizes
                                                fig, ax = plt.subplots(figsize=(8, 5))
                                                ax.bar(
                                                    segment_labels, 
                                                    segment_values
                                                )
                                                ax.set_xlabel("Segment")
                                                ax.set_ylabel("Number of Customers")
                                                ax.set_title("Segment Sizes")
                                                st.pyplot(fig)
                                                
                                                # Show sample IDs from each segment
                                                st.subheader("Sample Customer IDs by Segment")
                                                cluster_ids = {}
                                                if isinstance(cluster_stats, dict) and 'cluster_ids' in cluster_stats:
                                                    cluster_ids = cluster_stats['cluster_ids']
                                                for i in range(quick_segments):
                                                    if i in cluster_ids:
                                                        ids = cluster_ids[i]
                                                        sample_size = min(5, len(ids))
                                                        st.write(f"**Segment {i+1}:** {', '.join(map(str, ids[:sample_size]))}...")
                                                    else:
                                                        st.write(f"**Segment {i+1}:** No sample IDs available")
                                                
                                                # Info about detailed analysis
                                                st.info("For more detailed segment analysis, go to the 'Segment Analysis' tab.")
                                            else:
                                                st.error("Error generating segments.")
                                        except Exception as e:
                                            st.error(f"Error during segmentation: {str(e)}")
                                else:
                                    st.error("Not enough numeric features for clustering.")
                else:
                    st.warning("No customers match the selected criteria. Try adjusting your filters.")
        
        # Second tab - Clustering/Segmentation setup
        with tab2:
            st.header("Customer Segmentation")
            
            if len(filtered_df) < 2:
                st.warning("Not enough data for segmentation. Please adjust your filters to include more customers.")
            else:
                st.write("Use K-means clustering to segment customers based on selected attributes.")
                
                # Get numeric columns for clustering
                numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                # Let user select features for clustering
                selected_features = st.multiselect(
                    "Select features for segmentation",
                    numeric_cols,
                    default=st.session_state.selected_features if st.session_state.selected_features else ['age', 'income', 'purchase_amount', 'satisfaction_score']
                )
                st.session_state.selected_features = selected_features
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Let user select number of clusters/segments
                    n_clusters = st.slider(
                        "Number of segments",
                        min_value=2,
                        max_value=10,
                        value=st.session_state.n_clusters,
                        key="n_clusters_slider"
                    )
                    st.session_state.n_clusters = n_clusters
                
                with col2:
                    # Button to find optimal number of clusters
                    if st.button("Find Optimal Number of Segments", key="find_optimal"):
                        if len(selected_features) >= 2:
                            with st.spinner("Calculating optimal number of segments..."):
                                elbow_plot, message = clustering.get_optimal_clusters(
                                    filtered_df, 
                                    max_clusters=min(10, len(filtered_df) - 1), 
                                    numeric_columns=selected_features
                                )
                                
                                if elbow_plot:
                                    st.success("Analysis complete! See the elbow plot below.")
                                    st.image(f"data:image/png;base64,{elbow_plot}", caption="Elbow Method Plot")
                                    st.info("The 'elbow' in the graph shows the optimal number of segments. Select this number using the slider on the left.")
                                else:
                                    st.error(f"Error generating elbow plot: {message}")
                        else:
                            st.error("Please select at least 2 features for segmentation.")
                
                # Button to perform clustering
                if st.button("Generate Segments", key="generate_segments"):
                    if len(selected_features) >= 2:
                        with st.spinner("Generating customer segments..."):
                            # Apply k-means clustering
                            clustered_df, cluster_stats = clustering.apply_kmeans_clustering(
                                filtered_df, 
                                n_clusters=n_clusters,
                                numeric_columns=selected_features
                            )
                            
                            if clustered_df is not None:
                                st.session_state.clustered_data = clustered_df
                                st.session_state.cluster_stats = cluster_stats
                                st.success(f"Successfully created {n_clusters} customer segments!")
                                st.info("Go to the 'Segment Analysis' tab to view detailed information about each segment.")
                            else:
                                st.error(f"Error generating segments: {cluster_stats}")
                    else:
                        st.error("Please select at least 2 features for segmentation.")
        
        # Third tab - Analysis of segments
        with tab3:
            st.header("Segment Analysis")
            
            if st.session_state.clustered_data is None or st.session_state.cluster_stats is None:
                st.info("No segments generated yet. Go to the 'Customer Segmentation' tab to generate segments.")
            else:
                # Display segment information
                clustering.display_cluster_info(st.session_state.cluster_stats, filtered_df)
                
                # Feature comparison
                st.subheader("Feature Comparison")
                
                # Let user select features to compare
                numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                col1, col2 = st.columns(2)
                
                with col1:
                    x_feature = st.selectbox("X-axis feature", numeric_cols, index=numeric_cols.index('age') if 'age' in numeric_cols else 0)
                
                with col2:
                    y_feature = st.selectbox("Y-axis feature", numeric_cols, index=numeric_cols.index('income') if 'income' in numeric_cols else min(1, len(numeric_cols) - 1))
                
                if st.button("Generate Comparison Plot", key="gen_comparison"):
                    with st.spinner("Generating comparison plot..."):
                        scatter_plot, message = clustering.plot_cluster_comparison(
                            st.session_state.clustered_data,
                            x_feature,
                            y_feature
                        )
                        
                        if scatter_plot:
                            st.image(f"data:image/png;base64,{scatter_plot}", caption=f"{x_feature} vs {y_feature} by Segment")
                        else:
                            st.error(f"Error generating plot: {message}")
                
                # Download segmentation results
                st.subheader("Download Results")
                
                if st.session_state.clustered_data is not None:
                    csv_data = clustering.get_cluster_data_for_download(st.session_state.clustered_data)
                    href = f'<a href="data:file/csv;base64,{csv_data}" download="customer_segments.csv">Download Segment Data</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        # Fourth tab - Saved filters
        with tab4:
            st.header("Your Saved Filters")
            try:
                if st.session_state.user_id:
                    saved_filters = db.get_user_filters(st.session_state.user_id)
                    
                    if saved_filters:
                        filter_names = [name for name, _ in saved_filters]
                        selected_filter = st.selectbox("Select a saved filter", filter_names)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("Load Filter", key="load_filter_button"):
                                for name, filter_json in saved_filters:
                                    if name == selected_filter:
                                        st.session_state.current_filter = decode_filter_from_json(filter_json)
                                        st.session_state.active_tab = 0  # Switch to first tab
                                        st.rerun()
                        
                        with col2:
                            if st.button("Delete Filter", key="delete_filter_button"):
                                if selected_filter:
                                    success, message = db.delete_filter(st.session_state.user_id, selected_filter)
                                    if success:
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                    else:
                        st.info("You don't have any saved filters yet.")
                else:
                    st.info("Database unavailable. Filters cannot be saved in this session.")
            except Exception as e:
                st.warning("Could not retrieve saved filters. Database may be unavailable.")
                if st.session_state.show_debug_info:
                    st.error(f"Error: {str(e)}")

        # Sidebar with filtering options
        st.sidebar.title("Filters")
        
        # Add reset filters button
        if st.sidebar.button("Reset Filters", key="reset_filters"):
            st.session_state.current_filter = {}
            st.rerun()

        # Age filter
        age_min = int(st.session_state.data['age'].min())
        age_max = int(st.session_state.data['age'].max())
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=age_min,
            max_value=age_max,
            value=st.session_state.current_filter.get('age_range', (age_min, age_max)),
            key='age_filter'
        )
        st.session_state.current_filter['age_range'] = age_range

        # Income filter
        income_min = int(st.session_state.data['income'].min())
        income_max = int(st.session_state.data['income'].max())
        income_range = st.sidebar.slider(
            "Income Range ($)",
            min_value=income_min,
            max_value=income_max,
            value=st.session_state.current_filter.get('income_range', (income_min, income_max)),
            key='income_filter'
        )
        st.session_state.current_filter['income_range'] = income_range
        
        # Purchase Amount filter
        amount_min = int(st.session_state.data['purchase_amount'].min())
        amount_max = int(st.session_state.data['purchase_amount'].max())
        purchase_amount_range = st.sidebar.slider(
            "Purchase Amount Range ($)",
            min_value=amount_min,
            max_value=amount_max,
            value=st.session_state.current_filter.get('purchase_amount_range', (amount_min, amount_max)),
            key='purchase_amount_filter'
        )
        st.session_state.current_filter['purchase_amount_range'] = purchase_amount_range
        
        # Satisfaction Score filter
        score_min = int(st.session_state.data['satisfaction_score'].min())
        score_max = int(st.session_state.data['satisfaction_score'].max())
        satisfaction_range = st.sidebar.slider(
            "Satisfaction Score",
            min_value=score_min,
            max_value=score_max,
            value=st.session_state.current_filter.get('satisfaction_range', (score_min, score_max)),
            key='satisfaction_filter'
        )
        st.session_state.current_filter['satisfaction_range'] = satisfaction_range

        # Categorical filters with expanders for organization
        with st.sidebar.expander("Demographics", expanded=True):
            # Gender filter
            gender = st.selectbox(
                "Gender", 
                ["All"] + unique_values['gender'],
                index=0 if "gender" not in st.session_state.current_filter else 
                     (unique_values['gender'].index(st.session_state.current_filter['gender']) + 1),
                key='gender_filter'
            )
            if gender != "All":
                st.session_state.current_filter['gender'] = gender
            elif 'gender' in st.session_state.current_filter:
                del st.session_state.current_filter['gender']
                
            # Education filter
            education = st.selectbox(
                "Education", 
                ["All"] + unique_values['education'],
                index=0 if "education" not in st.session_state.current_filter else 
                     (unique_values['education'].index(st.session_state.current_filter['education']) + 1),
                key='education_filter'
            )
            if education != "All":
                st.session_state.current_filter['education'] = education
            elif 'education' in st.session_state.current_filter:
                del st.session_state.current_filter['education']
                
            # Region filter
            region = st.selectbox(
                "Region", 
                ["All"] + unique_values['region'],
                index=0 if "region" not in st.session_state.current_filter else 
                     (unique_values['region'].index(st.session_state.current_filter['region']) + 1),
                key='region_filter'
            )
            if region != "All":
                st.session_state.current_filter['region'] = region
            elif 'region' in st.session_state.current_filter:
                del st.session_state.current_filter['region']
        
        with st.sidebar.expander("Purchase Behavior", expanded=False):
            # Loyalty Status filter
            loyalty = st.selectbox(
                "Loyalty Status", 
                ["All"] + unique_values['loyalty_status'],
                index=0 if "loyalty_status" not in st.session_state.current_filter else 
                     (unique_values['loyalty_status'].index(st.session_state.current_filter['loyalty_status']) + 1),
                key='loyalty_filter'
            )
            if loyalty != "All":
                st.session_state.current_filter['loyalty_status'] = loyalty
            elif 'loyalty_status' in st.session_state.current_filter:
                del st.session_state.current_filter['loyalty_status']
                
            # Purchase Frequency filter
            frequency = st.selectbox(
                "Purchase Frequency", 
                ["All"] + unique_values['purchase_frequency'],
                index=0 if "purchase_frequency" not in st.session_state.current_filter else 
                     (unique_values['purchase_frequency'].index(st.session_state.current_filter['purchase_frequency']) + 1),
                key='frequency_filter'
            )
            if frequency != "All":
                st.session_state.current_filter['purchase_frequency'] = frequency
            elif 'purchase_frequency' in st.session_state.current_filter:
                del st.session_state.current_filter['purchase_frequency']
                
            # Product Category filter
            category = st.selectbox(
                "Product Category", 
                ["All"] + unique_values['product_category'],
                index=0 if "product_category" not in st.session_state.current_filter else 
                     (unique_values['product_category'].index(st.session_state.current_filter['product_category']) + 1),
                key='category_filter'
            )
            if category != "All":
                st.session_state.current_filter['product_category'] = category
            elif 'product_category' in st.session_state.current_filter:
                del st.session_state.current_filter['product_category']
                
            # Promotion Usage filter (boolean)
            promotion_options = ["All", "Yes", "No"]
            promotion_index = 0
            if 'promotion_usage' in st.session_state.current_filter:
                promotion_index = 1 if st.session_state.current_filter['promotion_usage'] == 1 else 2
                
            promotion = st.selectbox(
                "Used Promotion", 
                promotion_options,
                index=promotion_index,
                key='promotion_filter'
            )
            
            if promotion == "Yes":
                st.session_state.current_filter['promotion_usage'] = 1
            elif promotion == "No":
                st.session_state.current_filter['promotion_usage'] = 0
            elif 'promotion_usage' in st.session_state.current_filter:
                del st.session_state.current_filter['promotion_usage']

        # Logout button
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.current_filter = {}
            st.session_state.clustered_data = None
            st.session_state.cluster_stats = None
            st.rerun()

if __name__ == "__main__":
    main()