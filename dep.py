import streamlit as st
from rag_impl import generate_data_store, query_database

def main():
    st.title("RAG Interface")
    st.write("Enter your question in the text box below to query the document database.")
    
    query = st.text_input("Enter your question")
    if st.button("Submit"):
        if query:
            try:
                answer = query_database(query)
                st.write("Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
