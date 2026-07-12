import streamlit as st

def header_home():
    logo_url = "https://i.ibb.co/YTYGn5qV/logo.png"

    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 30px;
            margin-bottom: 30px;
        ">
            <img src="{logo_url}" width="100">
            <h1 style="
                text-align: center;
                color: #E0E3FF;
                margin-top: 10px;
            ">
                SNAP<br>CLASS
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )




