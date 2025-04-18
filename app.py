import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    if not text.strip():
        st.warning("No text could be extracted from the uploaded PDFs.")
    return text


def get_text_chunks(text):
    if not text.strip():
        st.warning("No text available to chunk.")
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning("No text chunks were created. Check the input text or chunking parameters.")
    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("Cannot create vector store: No text chunks provided.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversation_chain(vectorstore, model_choice):
    if vectorstore is None:
        st.error("Cannot create conversation chain: Invalid vector store.")
        return None

    model_options = {
        "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "DistilGPT2": "distilgpt2",
        "Qwen2-0.5B": "Qwen/Qwen2-0.5B-Instruct",
        "MobileLLM-125M": "facebook/MobileLLM-125M",
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct"
    }
    
    repo_id = model_options.get(model_choice, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        # Optional quantization for Phi-3-mini to reduce memory
        quantization_config = None
        if model_choice == "Phi-3-mini":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            quantization_config=quantization_config
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": None}
        )
        return conversation_chain

    except Exception as e:
        st.error(f"Failed to load model {model_choice}: {str(e)}")
        return None


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process documents first.")
        return
    if not user_question.strip():
        st.warning("Please enter a valid question.")
        return

    try:
        retriever = st.session_state.conversation.retriever
        docs = retriever.invoke(user_question)
        if not docs:
            st.warning("No relevant documents found for your question.")
            return

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        model_choice = st.selectbox(
            "Select LLM Model",
            ["TinyLlama", "DistilGPT2", "Qwen2-0.5B", "MobileLLM-125M", "Phi-3-mini"],
            index=0
        )
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("Failed to extract text from PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Failed to create text chunks.")
                        return
                    vectorstore = get_vectorstore(text_chunks)
                    if not vectorstore:
                        st.error("Failed to create vector store.")
                        return
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_choice)
                    if st.session_state.conversation:
                        st.success(f"Documents processed successfully with {model_choice} model!")
                    else:
                        st.error("Failed to initialize conversation chain. Please try another model.")
            else:
                st.warning("Please upload at least one PDF document.")


if __name__ == '__main__':
    main()