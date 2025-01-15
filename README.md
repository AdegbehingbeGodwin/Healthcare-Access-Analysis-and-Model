# Burkina Faso Healthcare Access Dashboard

This project provides an interactive dashboard for analyzing healthcare access patterns in Burkina Faso, focusing on factors such as regional disparities, socioeconomic status, education level, and more. It uses various machine learning models to predict healthcare access and offers valuable insights into the factors affecting healthcare delivery.

## Features

- **Geographic Pattern Detection**: Analyzes healthcare access rates across different regions.
- **Impact of Waiting Time**: Investigates the relationship between long waiting times and healthcare access.
- **Healthcare Access by Age Group**: Provides insights into how different age groups experience healthcare access.
- **Healthcare Access by Education Level**: Explores how education level correlates with healthcare access.
- **Predictive Model for Healthcare Access**: Uses a Random Forest model to predict healthcare access based on various input factors.

## Prerequisites

To run this project locally, you'll need the following software installed:

- Python 3.x
- Streamlit
- Plotly
- Pandas
- Scikit-learn
- Other dependencies as specified in the `requirements.txt` file.

### Required Libraries

- `streamlit`: For creating the interactive dashboard.
- `plotly`: For interactive visualizations.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For building and evaluating machine learning models.

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/burkina-faso-healthcare-dashboard.git
    cd burkina-faso-healthcare-dashboard
    ```

2. **Set up a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the dashboard**:

    ```bash
    streamlit run dashboard.py
    ```

The dashboard will be accessible at `http://localhost:8501` in your web browser.

## Data

The dashboard uses data from the "Community Monitoring for Better Health and Education Services Delivery Project" in Burkina Faso. The dataset contains various demographic and health-related factors, including household information, healthcare access, and barriers to service delivery.

### Key Columns:

- `region`: The region of the household in Burkina Faso.
- `hhreceiveservic`: A binary column indicating whether the household receives healthcare services (1 for Yes, 0 for No).
- `longwaitingtime`, `lackofnursingstaff`, `lackofdrugsupply`, etc.: Binary columns representing barriers to healthcare access (1 for Yes, 0 for No).
- `age`: The age of the household members.
- `educationlevel`: The education level of the head of the household.

## Features and Analysis

### 1. **Geographic Pattern Detection**
This section analyzes healthcare access rates by region and visualizes the disparities between different regions in Burkina Faso. It helps identify areas with lower access to healthcare services.

### 2. **Impact of Waiting Time on Healthcare Access**
This analysis examines the relationship between waiting times and healthcare access. It identifies how long waiting times affect the likelihood of receiving healthcare services.

### 3. **Healthcare Access by Age Group**
This section groups data by age and analyzes how healthcare access varies across different age groups, offering insights into which age groups may face more barriers.

### 4. **Healthcare Access by Education Level**
The dashboard explores how education level influences healthcare access, with insights into how improving education may improve healthcare outcomes.

### 5. **Predictive Model for Healthcare Access**
The dashboard includes a Random Forest model that predicts whether a household is likely to have healthcare access based on selected features. The model uses various demographic and healthcare-related data to make predictions.

## How to Use

- **Interactive Filters**: Use dropdown menus and text inputs to select features for prediction.
- **Visualizations**: Hover over the charts to view exact values, and use the interactive features to explore different aspects of healthcare access.
- **Model Predictions**: Input various features and click the "Predict Healthcare Access" button to receive a prediction of healthcare access for a specific household.

## Insights and Recommendations

The dashboard also provides key insights and actionable recommendations based on the data analysis, such as:

- Targeted interventions for regions with lower healthcare access.
- Age- and education-specific strategies for improving healthcare delivery.
- Addressing barriers like waiting times and nursing shortages to improve service uptake.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The data used in this project comes from the "Community Monitoring for Better Health and Education Services Delivery Project" in Burkina Faso.
- Special thanks to the contributors and data providers who made this project possible.
