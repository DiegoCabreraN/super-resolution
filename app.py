import streamlit as st

st.set_page_config(
   menu_items={}
)

st.logo('resources/logo.png')

st.markdown("""
   <style>
      .stLogo {
         height: 3.5rem;
      }

      .stDecoration {
         background-image: none;
         background-color: #2aa692;
      }
   </style>
""", unsafe_allow_html=True)

image_page = st.Page("pages/image_page.py", title="Image Upsampling", )
video_page = st.Page("pages/video_page.py", title="Video Upsampling")

pg = st.navigation([image_page, video_page])

pg.run()