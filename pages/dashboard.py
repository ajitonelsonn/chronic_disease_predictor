import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import get_saved_predictions
from datetime import datetime, timedelta
from components import navigation, footer
from styles import styles

# Modern color palette
COLORS = {
    'primary': '#6366f1',
    'secondary': '#818cf8',
    'success': '#22c55e',
    'warning': '#eab308',
    'danger': '#ef4444',
    'background': '#f8fafc',
    'card': '#ffffff',
    'text': '#1e293b',
    'subtext': '#64748b'
}

RISK_COLORS = {
    'High': COLORS['danger'],
    'Medium': COLORS['warning'],
    'Low': COLORS['success']
}

def set_page_style():
    """Configure page-wide styles"""
    st.set_page_config(
        page_title="Health Analytics | Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        
        [data-testid="stHeader"],
        #MainMenu,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] {
            display: none;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .filter-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
        }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border: 1px solid #e2e8f0;
        }
        
        .dataframe {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create a modern gradient header with subheader info"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="color: white; font-size: 2.75rem; font-weight: 700; margin: 0;">
            Patient Risk Analysis
        </h1>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; margin-top: 0.75rem;">
            Population Health Risk Assessment Dashboard
        </p>
    </div>
    
    <div style="
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    ">
        <p style="color: {COLORS['text']}; margin: 0; font-size: 0.95rem;">
            📊 <strong>Dashboard Overview:</strong> Monitor patient risk levels, analyze health trends, and track condition distributions. 
            Use the filters below to customize your view and identify high-risk patients requiring immediate attention.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
def add_section_info(title, description):
    """Add informative section headers"""
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    ">
        <h4 style="color: {COLORS['text']}; margin: 0 0 0.5rem 0; font-size: 1.1rem;">
            {title}
        </h4>
        <p style="color: {COLORS['subtext']}; margin: 0; font-size: 0.9rem;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, color=COLORS['text'], prefix="", suffix=""):
    """Create a modern metric card with hover effect"""
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: {COLORS['subtext']}; font-size: 1rem; margin: 0; 
                 font-weight: 500;">{title}</h4>
        <p style="color: {color}; font-size: 2.25rem; font-weight: 600; 
                margin: 0.5rem 0;">{prefix}{value}{suffix}</p>
    </div>
    """, unsafe_allow_html=True)

def filter_data(df, filters):
    """Filter dataframe based on selected criteria"""
    filtered_df = df.copy()
    
    if filters['gender'] != "All":
        filtered_df = filtered_df[filtered_df['gender'] == filters['gender']]
    if filters['race'] != "All":
        filtered_df = filtered_df[filtered_df['race'] == filters['race']]
    if filters['ethnicity'] != "All":
        filtered_df = filtered_df[filtered_df['ethnicity'] == filters['ethnicity']]
    if filters['condition'] != "All":
        filtered_df = filtered_df[
            filtered_df['conditions'].str.contains(filters['condition'], na=False)
        ]
    if filters['date_range']:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) & 
            (filtered_df['created_at'].dt.date <= end_date)
        ]
    
    return filtered_df

def create_trend_chart(df):
    """Create trend chart using only dates without time"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No data available for the selected filters',
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color='#64748b')
        )
    else:
        # Convert timestamp to date only and group by date and risk level
        daily_data = df.groupby([
            df['created_at'].dt.date,  # Extract date only
            'risk_level'
        ]).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for risk_level, color in {
            'High': '#ef4444',
            'Medium': '#eab308',
            'Low': '#22c55e'
        }.items():
            risk_data = daily_data[daily_data['risk_level'] == risk_level]
            if not risk_data.empty:
                fig.add_trace(go.Scatter(
                    x=risk_data['created_at'],
                    y=risk_data['count'],
                    name=f'{risk_level} Risk',
                    mode='lines+markers',
                    line=dict(
                        width=3,
                        color=color
                    ),
                    marker=dict(
                        size=8,
                        color=color
                    ),
                    hovertemplate=(
                        '<b>%{y}</b> patients<br>' +
                        '%{x|%Y-%m-%d}<br>' +
                        f'<b>{risk_level} Risk</b><extra></extra>'
                    )
                ))
    
    fig.update_layout(
        title='',
        xaxis_title=None,
        yaxis_title="Patient Count",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#e2e8f0',
            tickformat='%Y-%m-%d',  # Format x-axis labels as YYYY-MM-DD
            dtick='D1'  # Show tick for each day
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            showline=True,
            linecolor='#e2e8f0'
        )
    )
    
    return fig

def create_conditions_chart(df):
    """Create conditions distribution chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text='No data available',
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['subtext'])
        )
    else:
        conditions = df['conditions'].str.split(',').explode()
        top_conditions = conditions.value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=top_conditions.index,
            values=top_conditions.values,
            hole=0.7,
            marker=dict(colors=px.colors.sequential.Viridis),
            textinfo='label+percent',
            textposition='outside'
        )])
    
    fig.update_layout(
        title='',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        annotations=[dict(
            text='Primary<br>Conditions',
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False
        )]
    )
    
    return fig

def main():
    set_page_style()
    navigation()
    create_header()
    
    # Apply custom CSS
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    
    # Load data
    df = get_saved_predictions()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['confidence'] = pd.to_numeric(df['confidence']) * 100

    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox(
            "Gender",
            ["All"] + sorted(df['gender'].unique().tolist() if not df.empty else [])
        )
    with col2:
        race = st.selectbox(
            "Race",
            ["All"] + sorted(df['race'].unique().tolist() if not df.empty else [])
        )
    with col3:
        ethnicity = st.selectbox(
            "Ethnicity",
            ["All"] + sorted(df['ethnicity'].unique().tolist() if not df.empty else [])
        )
    
    col1, col2 = st.columns(2)
    with col1:
        conditions = (
            sorted(list(filter(None, df['conditions'].str.split(',').explode().unique())))
            if not df.empty else []
        )
        condition = st.selectbox("Condition", ["All"] + conditions)
    
    with col2:
        today = datetime.now()
        default_start = today - timedelta(days=30)
        date_range = st.date_input(
            "Date Range",
            value=(default_start.date(), today.date()),
            max_value=today.date()
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filters = {
        'gender': gender,
        'race': race,
        'ethnicity': ethnicity,
        'condition': condition,
        'date_range': date_range
    }
    filtered_df = filter_data(df, filters)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_patients = len(filtered_df['patient_id'].unique()) if not filtered_df.empty else 0
        create_metric_card("Total Patients", f"{total_patients:,}")
    
    with col2:
        high_risk = len(filtered_df[filtered_df['risk_level'] == 'High']) if not filtered_df.empty else 0
        create_metric_card(
            "High Risk Predictions",
            f"{high_risk:,}",
            color=COLORS['danger']
        )
    
    with col3:
        avg_confidence = filtered_df['confidence'].mean() if not filtered_df.empty else 0
        create_metric_card(
            "Average Confidence",
            f"{avg_confidence:.1f}",
            suffix="%"
        )
    
    # Charts
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_section_info(
            "Risk Level Analysis",
            "Visualizes patient risk distribution over time. Track trends to identify patterns and potential areas requiring intervention."
        )
        st.plotly_chart(
            create_trend_chart(filtered_df),
            use_container_width=True,
            config={'displayModeBar': False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        add_section_info(
            "Health Conditions Overview",
            "Shows distribution of primary health conditions across the patient population. Identify prevalent conditions requiring focused care strategies."
        )
        st.plotly_chart(
            create_conditions_chart(filtered_df),
            use_container_width=True,
            config={'displayModeBar': False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data viewer
    with st.expander("View Raw Data"):
        if not filtered_df.empty:
            display_df = filtered_df.copy()
            display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['confidence'] = display_df['confidence'].round(2).astype(str) + '%'
            
            st.dataframe(
                display_df[[
                    'patient_id', 'age', 'gender', 'race', 'ethnicity',
                    'primary_condition', 'conditions', 'risk_level',
                    'confidence', 'created_at'
                ]],
                use_container_width=True,
                height=400
            )
        else:
            st.info("No data available for the selected filters")
    
    footer()

if __name__ == "__main__":
    main()