import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import os
import traceback
import time

# ----- Page Config -----
st.set_page_config(page_title="Play with RAG", page_icon="📚", layout="centered")

# ----- Custom CSS -----
st.markdown("""
    <style>
        .title {
            font-size: 3em;
            font-weight: 700;
            color: #4B8BBE;
            margin-bottom: 0.2em;
        }
        .subtitle {
            font-size: 1.2em;
            color: #6c757d;
            margin-bottom: 2em;
        }
        .sidebar-text {
            font-size: 0.95em;
            line-height: 1.6;
        }
        .sidebar-link a {
            color: #4B8BBE !important;
            text-decoration: none;
        }
        .sidebar-link a:hover {
            text-decoration: underline;
        }
        .stProgress > div > div > div > div {
            background-color: #4B8BBE;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Sidebar -----
# Keep the original sidebar configuration for API key and model selection
with st.sidebar:
    # try:
    #     st.image("The ghibli.jpg", width=140, caption="Md Faisal Karim")
    # except:
    #     st.markdown("## Md Faisal Karim")
    
    # st.markdown('<div class="sidebar-text">👨‍💻 <b>AI Researcher | CSE Graduate</b><br>📍 Khulna, Bangladesh</div>', unsafe_allow_html=True)
    
    # st.markdown("---")
    
    # Configuration section
    st.header("App Configuration")
    
    # API key input
    api_key = st.text_input("OpenRouter API Key", type="password", 
                           help="Enter your OpenRouter API key")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["deepseek/deepseek-r1:free", "anthropic/claude-3-opus", "google/gemini-1.5-pro"],
        help="Choose the model to use for generating responses"
    )


        # Model selection
    model = st.selectbox(
        "Select Embedding Model",
        ["all-MiniLM-L6-v2", "Gemma", "Ollama"],
        help="Choose the model to use for generating embeddings"
    )
    
    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 100, 1000, 500, 
                              help="Size of text chunks for processing")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.9, 0.1,
                               help="Controls randomness of output")
        max_tokens = st.slider("Max Tokens", 100, 10000, 5000, 500,
                              help="Maximum length of generated response")
        k = st.slider("Results to retrieve", 1, 10, 5,
                     help="Number of document chunks to retrieve")
    
    st.markdown("---")
    
    # Contact info
    st.markdown('<div class="sidebar-text sidebar-link">🔗 <a href="#" target="_blank">Portfolio Website</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text sidebar-link">🐱 <a href="https://github.com/Faisalkarim" target="_blank">GitHub</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text sidebar-link">💼 <a href="https://www.linkedin.com/in/itsfaisalkarim/" target="_blank">LinkedIn</a></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="sidebar-text">📧 itsfaisalkarim@gmail.com </div>', unsafe_allow_html=True)

# ----- Title -----
st.markdown('<div class="title">Ask Me About your documents</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive techical aspect focused RAG. Upload your docs & do not forget to enter API key</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
    st.session_state.chunk_sources = []
    st.session_state.index = None
    st.session_state.embeddings = None
    st.session_state.model = None
    st.session_state.files_processed = False

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {file.name}: {str(e)}")
        return ""

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ----- File Upload Section -----
if not st.session_state.files_processed:
    with st.expander("📄 Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        col1, col2, col3 = st.columns([3.3, 2, 3])
        with col2:
            process_button = st.button("Process Files", type="primary")
        
        if uploaded_files and process_button and not st.session_state.files_processed:
            try:
                st.session_state.all_chunks = []
                st.session_state.chunk_sources = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each file
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i) / len(uploaded_files))
                    
                    # Extract text from PDF
                    raw_text = extract_text_from_pdf(file)
                    if not raw_text:
                        continue
                        
                    # Chunk the extracted text
                    chunks = chunk_text(raw_text, chunk_size)
                    
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.chunk_sources.extend([file.name] * len(chunks))
                    
                    progress_bar.progress((i + 0.5) / len(uploaded_files))
                
                if st.session_state.all_chunks:
                    # Generate embeddings
                    status_text.text("Generating embeddings...")
                    progress_bar.progress(0)
                    # Gradually fill progress bar to 0.9
                    for percent_complete in range(80):  # 50 to 90 inclusive
                        time.sleep(0.02)  # Adjust speed here
                        progress_bar.progress(percent_complete +1)
                        
                    try:
                        # Load the model once and store in session state
                        if st.session_state.model is None:
                            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                        embeddings = st.session_state.model.encode(st.session_state.all_chunks, show_progress_bar=False)
                        st.session_state.embeddings = embeddings
                        
                        # Index embeddings using FAISS
                        status_text.text("Indexing embeddings...")
                        progress_bar.progress(0.9)
                        time.sleep(0.05)  # Adjust speed here

                        embedding_dim = embeddings.shape[1]
                        index = faiss.IndexFlatL2(embedding_dim)
                        index.add(np.array(embeddings))
                        st.session_state.index = index
                        
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        st.session_state.files_processed = True
                        
                        # Display success message
                        st.success(f"Successfully processed {len(uploaded_files)} PDFs with {len(st.session_state.all_chunks)} total chunks")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"I've processed all({len(uploaded_files)}) the documents. You can now ask me questions about them!"
                        })
                        
                    except Exception as e:
                        st.error(f"Error during embedding or indexing: {str(e)}")
                        st.code(traceback.format_exc())
                else:
                    st.warning("No text chunks were extracted from the PDFs.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.code(traceback.format_exc())

# ----- Optional: Paper List -----
with st.expander("📄 View Documents"):
    if st.session_state.files_processed:
        # Show uploaded papers
        file_counts = {}
        for source in st.session_state.chunk_sources:
            if source not in file_counts:
                file_counts[source] = 0
            file_counts[source] += 1
        
        for file, count in file_counts.items():
            st.markdown(f"- **{file}** – *{count} text chunks*")
    # else:
    #     st.markdown("""
    #     - **Early Detection of Glaucoma from Fundus Images** – *IEEE Xplore, 2024*  
    #     - **Dual-Channel CNN for COVID-19 Detection** – *Conference Paper*  
    #     - **Brain Tumor Classification from MRI** – *Ongoing Work*  
    #     """)

# ----- Chat Interface -----
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input area and processing
prompt = st.chat_input("Ask a question about documents...")
if prompt and st.session_state.files_processed:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the query
    try:
        with st.spinner("Thinking..."):
            # Generate query embedding
            query_embedding = st.session_state.model.encode([prompt])
            
            # Search for similar chunks
            distances, indices = st.session_state.index.search(np.array(query_embedding), k)
            
            # Prepare context from retrieved chunks
            context_chunks = []
            for i, idx in enumerate(indices[0]):
                source = st.session_state.chunk_sources[idx]
                text = st.session_state.all_chunks[idx]
                context_chunks.append(f"[From {source}]\n{text}")
            
            context = "\n\n".join(context_chunks)
            
            # Check if API key is provided
            if not api_key:
                with st.chat_message("assistant"):
                    st.markdown("⚠️ Please enter your OpenRouter API key in the sidebar to generate responses.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ Please enter your OpenRouter API key in the sidebar to generate responses."
                })
            else:
                # Generate response using selected model
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Prepare prompt
                system_prompt = """You are a helpful RAG system. Answer the user's question based on the 
                provided context from documents and also use your own understanding saying that, 'I think'. 
                If the answer cannot be found in the context, use your own knowledge and use similarity with users context.
                If you think, user's prompt is completely nowhere near to the context, ask politely to ask relevent questions. 
                Include references to the specific documents & other web resources u gonna use, when appropriate.
                """
                
                user_prompt = f"""Question: {prompt}
                
                Context from relevant documents:
                {context}
                
                Please provide a helpful, accurate answer based only on the information in these documents."""
                
                data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions", 
                        headers=headers, 
                        data=json.dumps(data),
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer
                        })
                    else:
                        with st.chat_message("assistant"):
                            st.error(f"API request failed with status code {response.status_code}")
                            st.code(response.text)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"⚠️ API request failed with status code {response.status_code}. Please try again."
                        })
                except requests.exceptions.RequestException as e:
                    with st.chat_message("assistant"):
                        st.error(f"API request error: {str(e)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"⚠️ API request error: {str(e)}. Please try again."
                    })
        
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error processing your question: {str(e)}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"⚠️ Error processing your question: {str(e)}. Please try again."
        })

elif prompt and not st.session_state.files_processed:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        st.markdown("Please upload and process some documents first before asking questions.")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Please upload and process some documents first before asking questions."
    })

# Instructions for first-time users
if not st.session_state.files_processed and not st.session_state.messages:
    # with st.chat_message("assistant"):
    #     st.markdown("""
    #     👋 Welcome! What's cooking???
        
    #     To get started:
    #     1. Upload your documents using the uploader above
    #     2. Click "Process Files" to analyze them
    #     3. Ask me questions about your documents in the chat input below
        
    #     You can change the settings and play around with the model and parameters in the sidebar.
    #     I'll use RAG (Retrieval-Augmented Generation) to find relevant information in your documents and provide accurate answers.
    #     """)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """
        👋 Welcome! What's cooking???
        
        Now,
        1. Click "Process Files" to analyze them
        2. Ask me questions about your documents in the chat input below
        
        You can change the settings and play around with the model and parameters in the sidebar.
        RAG (Retrieval-Augmented Generation) gonna find relevant information in your documents and provide accurate answers.
        """
    })