import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration
st.set_page_config(layout="wide", page_title="Burkina Faso Healthcare Access Dashboard")

# Function to load and prepare data
@st.cache_data
def load_data():
    # Load your CSV file
    df = pd.read_csv('output.csv')  # Replace with your actual file path
    
    # Convert "yes"/"no" columns to numeric values
    yes_no_columns = ['longwaitingtime', 'lackofnursingstaff', 'lackofrespect', 
                      'lackofdrugsupply', 'tooexpenscantpay', 'someonehhneedcsps', 
                      'hhreceiveservic']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
    
    return df

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Dashboard title
st.title("🏥 Healthcare Access Analysis Dashboard - Burkina Faso")
st.markdown("### Comprehensive Analysis of Healthcare Barriers and Demographics")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_households = df['household'].nunique() if 'household' in df.columns else len(df)
    st.metric("Total Households Surveyed", f"{total_households:,}")

with col2:
    avg_household_size = df['peoplivinghh'].mean() if 'peoplivinghh' in df.columns else 0
    st.metric("Average Household Size", f"{avg_household_size:.1f}")

with col3:
    if 'hhreceiveservic' in df.columns:
        service_access_rate = df['hhreceiveservic'].mean() * 100
    else:
        service_access_rate = 0
    st.metric("Healthcare Access Rate", f"{service_access_rate:.1f}%")

with col4:
    if 'lackofnursingstaff' in df.columns:
        nursing_shortage = df['lackofnursingstaff'].mean() * 100
    else:
        nursing_shortage = 0
    st.metric("Nursing Shortage Rate", f"{nursing_shortage:.1f}%")



# Socioeconomic Analysis
st.markdown("## 💰 Socioeconomic Impact")
col1, col2 = st.columns(2)

with col1:
    if 'mainsourcehhincome' in df.columns:
        income_access = df.groupby('mainsourcehhincome')['hhreceiveservic'].mean().sort_values() * 100

        fig = go.Figure(go.Bar(
            x=income_access.values,
            y=income_access.index,
            orientation='h',
            text=income_access.round(1).astype(str) + '%'
        ))
        fig.update_layout(title="Healthcare Access by Income Source", height=400)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'religioncm' in df.columns:
        religion_access = df.groupby('religioncm')['hhreceiveservic'].mean() * 100

        fig = px.pie(
            values=religion_access.values,
            names=religion_access.index,
            title="Healthcare Access by Religion"
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Regional Analysis
st.markdown("### 📊 Regional Healthcare Analysis")
col1, col2 = st.columns(2)

with col1:
    if 'region' in df.columns:
        # Regional Distribution of Healthcare Barriers
        barrier_columns = [col for col in ['lackofnursingstaff', 'lackofdrugsupply', 
                                           'longwaitingtime', 'tooexpenscantpay']
                           if col in df.columns]

        barriers_by_region = pd.DataFrame({
            col.replace('lackof', '').replace('too', '').title(): df.groupby('region')[col].mean()
            for col in barrier_columns
        }) * 100

        fig = go.Figure()
        for column in barriers_by_region.columns:
            fig.add_trace(go.Bar(
                name=column,
                x=barriers_by_region.index,
                y=barriers_by_region[column],
                text=barriers_by_region[column].round(1).astype(str) + '%'
            ))

        fig.update_layout(
            title="Healthcare Barriers by Region",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if all(col in df.columns for col in ['age', 'sex', 'hhreceiveservic']):
        # Service Access by Demographics
        df['age_group'] = pd.cut(df['age'],
                                bins=[0, 15, 30, 45, 60, 100],
                                labels=['0-15', '16-30', '31-45', '46-60', '60+'])

        access_by_age_gender = df.groupby(['age_group', 'sex'])['hhreceiveservic'].mean().unstack()

        fig = go.Figure()
        for gender in access_by_age_gender.columns:
            fig.add_trace(go.Bar(
                name=gender,
                x=access_by_age_gender.index,
                y=access_by_age_gender[gender] * 100,
                text=(access_by_age_gender[gender] * 100).round(1).astype(str) + '%'
            ))

        fig.update_layout(
            title="Healthcare Access by Age and Gender",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)



# Geographic Pattern Detection
st.markdown("### 🌍 Geographic Pattern Detection by Region")

if 'region' in df.columns and 'hhreceiveservic' in df.columns:
    # Regional Service Access Rate
    region_access = df.groupby('region')['hhreceiveservic'].mean() * 100

    # Plotting the regional data
    fig = go.Figure(go.Bar(
        x=region_access.index,
        y=region_access.values,
        text=region_access.round(1).astype(str) + '%',
        marker=dict(color=region_access.values, colorscale='Viridis')
    ))

    fig.update_layout(
        title="Healthcare Access by Region",
        xaxis_title="Region",
        yaxis_title="Access Rate (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Insights")
    st.write("Regions with lower healthcare access rates may require targeted interventions.")

# Correlation Analysis
st.markdown("### 🔄 Healthcare Barrier Correlation Analysis")
barrier_columns = [col for col in ['longwaitingtime', 'lackofnursingstaff', 'lackofrespect',
                                   'lackofdrugsupply', 'tooexpenscantpay']
                   if col in df.columns]

if barrier_columns:
    correlation_matrix = df[barrier_columns].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=[col.replace('lackof', '').replace('too', '').title() for col in barrier_columns],
        y=[col.replace('lackof', '').replace('too', '').title() for col in barrier_columns],
        text=correlation_matrix.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))

    fig.update_layout(
        title="Correlation between Healthcare Barriers",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Add insights and recommendations
st.markdown("### 🔍 Key Insights")
st.write("""
- **Regional Disparities**: Significant variations in healthcare access across regions
- **Demographic Patterns**: Age and gender play crucial roles in healthcare access
- **Socioeconomic Impact**: Income source strongly correlates with healthcare access
- **Barrier Correlations**: Strong relationships between different healthcare barriers
""")

st.markdown("### 📋 Recommendations")
st.write("""
1. **Target Resource Allocation**: Focus on regions with highest barrier rates
2. **Demographic-Specific Programs**: Develop targeted interventions for underserved age-gender groups
3. **Economic Support**: Implement financial assistance programs for vulnerable income groups
4. **Integrated Solutions**: Address correlated barriers through comprehensive programs
""")
# Predictive Modeling Section
st.markdown("### 🔮 Predictive Modeling for Healthcare Access")

if 'hhreceiveservic' in df.columns:
    # Feature Selection
    feature_columns = ['longwaitingtime', 'lackofnursingstaff', 'lackofrespect', 
                       'lackofdrugsupply', 'tooexpenscantpay', 'peoplivinghh']
    feature_columns = [col for col in feature_columns if col in df.columns]

    X = df[feature_columns]
    y = df['hhreceiveservic']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Model Metrics
    #st.write(f"**Model Accuracy**: {accuracy * 100:.2f}%")
    #st.text("Classification Report:")
    #st.text(classification_report(y_test, y_pred))

    # Feature Importance
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance", labels={'Importance': 'Importance'})
    st.plotly_chart(fig, use_container_width=True)

    # Prediction Tool
    st.markdown("#### 🎯 Make a Prediction")
    input_data = {}
    for col in feature_columns:
        if col in ['longwaitingtime', 'lackofnursingstaff', 'lackofrespect', 
                   'lackofdrugsupply', 'tooexpenscantpay']:
            input_data[col] = st.selectbox(f"Select value for {col}", options=['Yes', 'No'])
        else:
            input_data[col] = st.number_input(f"Enter value for {col} (e.g., household size)", 
                                              min_value=0.0, step=1.0)

    # Convert dropdown selections to numeric
    for key, value in input_data.items():
        if value == 'Yes':
            input_data[key] = 1
        elif value == 'No':
            input_data[key] = 0

    if st.button("Predict Healthcare Access"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1]

        st.write(f"**Prediction**: {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability of Access**: {prediction_prob * 100:.2f}%")
else:
    st.warning("Healthcare access column ('hhreceiveservic') is missing from the dataset.")


