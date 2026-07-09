import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("Customer_Segmentation_100K.xlsx")
    return df

df = load_data()

# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("🛍️ Customer Segmentation")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Visualization",
        "🤖 Predict Customer",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Machine Learning Project

Algorithm:
K-Means Clustering

Dataset:
100,000 Customers
"""
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🛍️ Customer Segmentation Dashboard")

    st.write(
        """
        Welcome to the Customer Segmentation Dashboard.

        This application uses the **K-Means Clustering**
        algorithm to classify customers based on:

        • Annual Spending

        • Orders Count
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}"
        )

    with col2:
        st.metric(
            label="Average Spending",
            value=f"₹ {int(df['Annual_Spending'].mean()):,}"
        )

    with col3:
        st.metric(
            label="Average Orders",
            value=round(df["Orders_Count"].mean(),2)
        )

    st.markdown("---")

    st.subheader("Dataset Overview")

    st.dataframe(df.head(10), use_container_width=True)

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "📂 Dataset":

    st.title("📂 Dataset")

    st.subheader("Dataset Shape")

    rows, cols = df.shape

    c1, c2 = st.columns(2)

    c1.metric("Rows", rows)

    c2.metric("Columns", cols)

    st.markdown("---")

    st.subheader("First 20 Records")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe(), use_container_width=True)

    # =====================================================
# VISUALIZATION PAGE
# =====================================================

elif page == "📊 Visualization":

    st.title("📊 Customer Data Visualization")

    st.markdown("Explore the customer dataset using interactive charts.")

    st.markdown("---")

    # Histogram Row
    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(
            df,
            x="Annual_Spending",
            nbins=40,
            title="Annual Spending Distribution",
            color_discrete_sequence=["royalblue"]
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            df,
            x="Orders_Count",
            nbins=30,
            title="Orders Count Distribution",
            color_discrete_sequence=["green"]
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Scatter Plot
    fig3 = px.scatter(
        df,
        x="Annual_Spending",
        y="Orders_Count",
        title="Annual Spending vs Orders Count",
        opacity=0.6,
        color="Orders_Count"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Box Plots
    c1, c2 = st.columns(2)

    with c1:

        fig4 = px.box(
            df,
            y="Annual_Spending",
            title="Annual Spending Box Plot",
            color_discrete_sequence=["orange"]
        )

        st.plotly_chart(fig4, use_container_width=True)

    with c2:

        fig5 = px.box(
            df,
            y="Orders_Count",
            title="Orders Count Box Plot",
            color_discrete_sequence=["purple"]
        )

        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    st.subheader("Correlation Matrix")

    corr = df[["Annual_Spending", "Orders_Count"]].corr()

    fig6 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Feature Correlation"
    )

    st.plotly_chart(fig6, use_container_width=True)

    # =====================================================
# PREDICT CUSTOMER PAGE
# =====================================================

elif page == "🤖 Predict Customer":

    st.title("🤖 Customer Segmentation Prediction")

    st.write(
        """
        Enter the customer's details below to predict
        which segment they belong to.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        spending = st.number_input(
            "Annual Spending (₹)",
            min_value=1000,
            max_value=100000,
            value=25000,
            step=1000
        )

    with col2:

        orders = st.number_input(
            "Orders Count",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

    st.markdown("")

    if st.button("Predict Customer Segment", use_container_width=True):

        # Prepare input
        customer = np.array([[spending, orders]])

        customer_scaled = scaler.transform(customer)

        cluster = model.predict(customer_scaled)[0]

        # Get cluster centers in original scale
        centers = scaler.inverse_transform(model.cluster_centers_)

        spending_centers = centers[:, 0]

        # Sort clusters by spending
        sorted_clusters = np.argsort(spending_centers)

        low_cluster = sorted_clusters[0]
        medium_cluster = sorted_clusters[1]
        high_cluster = sorted_clusters[2]

        st.markdown("---")

        st.subheader("Prediction Result")

        if cluster == high_cluster:

            st.success("💎 High Value Customer")

            st.write("""
            ✔ Frequently purchases products

            ✔ High annual spending

            ✔ Premium customer

            ✔ Suitable for loyalty rewards
            """)

        elif cluster == medium_cluster:

            st.info("⭐ Medium Value Customer")

            st.write("""
            ✔ Regular customer

            ✔ Moderate spending

            ✔ Potential premium customer
            """)

        else:

            st.warning("🛒 Low Value Customer")

            st.write("""
            ✔ Low spending

            ✔ Fewer purchases

            ✔ Marketing campaigns recommended
            """)

        st.markdown("---")

        c1, c2 = st.columns(2)

        c1.metric("Annual Spending", f"₹ {spending:,}")

        c2.metric("Orders Count", orders)