import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

# Main App
def main():
    # Load the dataset
    data = load_data()

    st.title("Heart Disease Analysis Dashboard")
    st.write("Visualize the impact of **Age**, **Sex**, and **Resting Blood Pressure (BP)** on heart disease.")

    # User Inputs
    st.sidebar.header("Filter Options")

    # Age Filter
    age_filter = st.sidebar.radio("Age Filter", options=["Individual Age", "Group Age Range"])
    if age_filter == "Individual Age":
        selected_age = st.sidebar.number_input("Enter Age", min_value=int(data["age"].min()), max_value=int(data["age"].max()), value=30)
        filtered_data = data[data["age"] == selected_age]
    else:
        age_range = st.sidebar.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (30, 50))
        filtered_data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

    # Sex Filter
    sex_filter = st.sidebar.radio("Sex Filter", options=["Both", "Male", "Female"])
    if sex_filter != "Both":
        sex_value = 1 if sex_filter == "Female" else 0
        filtered_data = filtered_data[filtered_data["sex"] == sex_value]

    # Blood Pressure Filter
    bp_range = st.sidebar.slider("Resting BP (trestbps) Range", int(data["trestbps"].min()), int(data["trestbps"].max()), (120, 140))
    filtered_data = filtered_data[(filtered_data["trestbps"] >= bp_range[0]) & (filtered_data["trestbps"] <= bp_range[1])]

    # Display Filters Summary
    st.write("### Filter Summary")
    st.write(f"**Age**: {age_filter}")
    if age_filter == "Individual Age":
        st.write(f"Selected Age: {selected_age}")
    else:
        st.write(f"Age Range: {age_range[0]} - {age_range[1]}")
    st.write(f"**Sex**: {sex_filter}")
    st.write(f"**BP Range**: {bp_range[0]} - {bp_range[1]}")

    # Display Filtered Data
    st.write("### Filtered Data")
    st.write(filtered_data)

    # Visualization Section
    st.write("### Visualizations")
    st.write("Below are graphs generated based on the filtered data:")

    # Age Distribution
    st.write("#### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data["age"], bins=10, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # BP Distribution
    st.write("#### Resting BP Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x="sex", y="trestbps", data=filtered_data, ax=ax)
    ax.set_title("Resting BP Distribution by Sex")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Male", "Female"])
    st.pyplot(fig)

    # Scatter Plot: Age vs. BP
    st.write("#### Age vs. Resting BP")
    fig, ax = plt.subplots()
    sns.scatterplot(x="age", y="trestbps", hue="target", data=filtered_data, ax=ax, palette="coolwarm")
    ax.set_title("Age vs. Resting BP (Colored by Target)")
    st.pyplot(fig)

    # Relationship Between Age, BP, and Heart Disease
    st.write("#### Age vs. BP Grouped by Heart Disease")
    fig, ax = plt.subplots()
    sns.violinplot(x="target", y="age", hue="sex", data=filtered_data, split=True, palette="Set2", ax=ax)
    ax.set_title("Age Distribution Grouped by Heart Disease and Sex")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Heart Disease", "Heart Disease"])
    st.pyplot(fig)

    # Show warning if filtered data is empty
    if filtered_data.empty:
        st.warning("No data matches the current filters. Please adjust the filters.")

if __name__ == "__main__":
    main()
