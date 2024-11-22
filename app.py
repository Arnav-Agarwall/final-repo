import streamlit as st
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from io import BytesIO
from docx import Document
from fpdf import FPDF
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate



st.set_page_config(
    page_title="LLM Text Summarizer",
    layout="centered",
    initial_sidebar_state="auto",
)


st.title("LLM-Based Text Summarizer")


st.markdown("""
Enter the text you want to summarize below.
""")


input_text = st.text_area("Enter Text to Summarize:", height=300)


llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", huggingfacehub_api_token="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")


prompt = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary for the following text:\n\n{text}\n\nSummary:",
)


chain = LLMChain(llm=llm, prompt=prompt)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            summary = chain.run({"text": input_text})
        st.success("Summary Generated!")
        st.text_area("Summary:", value=summary, height=200)

        
        st.markdown("### Download Summary:")

        
        def get_txt_file(content):
            return BytesIO(content.encode('utf-8'))

        
        def get_docx_file(content):
            doc = Document()
            doc.add_paragraph(content)
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer

        def get_pdf_file(content):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in content.split('\n'):
                pdf.multi_cell(0, 10, line)
            buffer = BytesIO()
            pdf.output(buffer, "S")  # Pass "S" as the destination to write to a string buffer
            buffer.seek(0)
            return buffer


        
        txt_file = get_txt_file(summary)
        docx_file = get_docx_file(summary)
        pdf_file = get_pdf_file(summary)

        
        st.download_button(
            label="Download as TXT",
            data=txt_file,
            file_name="summary.txt",
            mime="text/plain",
        )

        st.download_button(
            label="Download as DOCX",
            data=docx_file,
            file_name="summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        st.download_button(
            label="Download as PDF",
            data=pdf_file,
            file_name="summary.pdf",
            mime="application/pdf",
        )
