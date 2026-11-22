import textwrap

import streamlit as st

from agent import build_app, ticker_check

MODEL = "llama-3.1-8b-instant"

app = build_app(MODEL)


def write_wrapped(text: str):
    for line in text.split("\n"):
        st.write(textwrap.fill(line, 80))


st.title("🤖 Crypto Financial Advisor")

st.subheader("Enter Your Query")
user_query = st.text_input("Ask your financial question (e.g., 'BTC price analysis')")

if st.button("Get Report"):
    if not user_query:
        st.error("Please enter a query.")
    else:
        state = app.invoke({"user_query": user_query})

        if ticker_check(state) == "yes":
            st.subheader("📈 Price Analyst Report")
            write_wrapped(state.get("price_analyst_report", "No report available"))

            st.subheader("📰 News Analyst Report")
            write_wrapped(state.get("news_analyst_report", "No report available"))

            report = state.get("final_report")
            if report:
                st.subheader("📊 Final Report")
                st.write(f"**Action:** {report.action}")
                st.write(f"**Score:** {report.score}")
                st.write(f"**Trend:** {report.trend}")
                st.write(f"**Sentiment:** {report.sentiment}")

                st.subheader("🔮 Price Predictions (4 Weeks Ahead)")
                st.write(report.price_predictions)

                st.subheader("📃 Summary")
                st.write(textwrap.fill(report.summary))
                st.subheader("Final Answer")
                st.write(state["final_response"][-1].content)
            else:
                st.error("No final report available.")
        else:
            st.subheader("Final Answer")
            st.write(state["final_response"][-1].content)

st.markdown("---")
st.markdown("Developed by Sambhav Sharma with ❤️")
