def main():
    st.set_page_config(page_title="Mezyan", page_icon=":palm_tree:", layout="wide")

    st.title("Mezyan")
    st.image("Mezyan Logo.png", width=200)
    st.subheader("South Mediterranean trattoria and café")
    st.write("Serving a range of Lebanese, Armenian, Moroccan dishes and local wines.")
    st.write("Address: Beirut, Hamra Street, Rasamny building")
    st.write("Phone: 71 293 015")
    st.write("""
    **Hours:**
    - Sunday to Wednesday: 9:30 AM – 12:30 AM
    - Thursday to Saturday: 9:30 AM – 1:30 AM
    """)

    with st.sidebar:
        tab = st.tabs(["Home", "Customer Feedback", "Waste Reduction"])

    if tab == "Home":
        st.write("Welcome to Mezyan!")
    elif tab == "Customer Feedback":
        st.subheader("Customer Feedback")
        # Implement the functionality to collect customer feedback
        feedback = st.text_area("Please share your experience with us:")
        submit_button = st.button("Submit Feedback")
        if submit_button:
            st.success("Thank you for your feedback!")
    elif tab == "Waste Reduction":
        st.subheader("Waste Reduction Initiatives")
        st.write("Learn more about our efforts to reduce waste and how you can contribute.")
