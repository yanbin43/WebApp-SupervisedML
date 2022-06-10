import streamlit as st

st.title('Welcome!')

content = '''
          This is a machine learning demo with **Linear Regression** and **Classification**.
          <br><br>
          Developed by Tan Yan Bin
          <br>
          Contact me at [LinkedIn](https://www.linkedin.com/in/yyanbin-tan/) | [E-mail](mail.to:yybbb0001@gmail.com)
          '''

st.markdown(content, unsafe_allow_html = True)
