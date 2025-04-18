# KNOW-YOUR-DOCS
# 📚 Chat with Multiple PDFs using Local LLMs (Streamlit + LangChain)

A powerful and lightweight app that allows users to **upload multiple PDF documents** and **ask questions about their contents** using **local LLMs** like TinyLlama, Phi-3, Qwen, and more — all from a simple **Streamlit interface**.

---

## 🚀 Features

- ✅ Upload and query **multiple PDF documents**
- ✅ Select from a range of **open-source LLMs**
- ✅ Embedding-based semantic search using **FAISS**
- ✅ Retains **conversational memory** for contextual responses
- ✅ Runs **fully offline** — no external API needed
- ✅ Built with **Streamlit, LangChain, Hugging Face, and PyPDF2**

---

## 🛠️ Tech Stack

| Tool / Library         | Purpose                                      |
|------------------------|----------------------------------------------|
| Streamlit              | Web Interface                                |
| LangChain              | Conversational memory + LLM chaining         |
| HuggingFace Transformers | LLMs and tokenizers                       |
| HuggingFace Embeddings | Text vectorization                           |
| FAISS                  | Fast vector-based similarity search          |
| PyPDF2                 | Extract text from PDFs                       |
| dotenv                 | Manage environment variables (optional)      |

---

## 📂 Project Structure

2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file if needed:

env
Copy
Edit
HUGGINGFACEHUB_API_TOKEN=your_token_here  # Optional
4. Run the App
bash
Copy
Edit
streamlit run app.py
🤖 Supported LLM Models

Model	Description
TinyLlama	Lightweight, fast-performing open model
DistilGPT2	Compact version of GPT2
Qwen2-0.5B	Alibaba's efficient instruction-tuned model
MobileLLM-125M	Meta’s mobile-friendly language model
Phi-3-mini	Microsoft’s compact, high-performance model
💡 How It Works
📄 Upload one or more PDF documents

🤖 Choose an LLM model from the sidebar

⚙️ Click “Process” to extract text and create vector store

💬 Ask any question about your documents

🧠 Get intelligent, context-aware responses from your selected model

📌 Notes
Some models may require substantial memory — use smaller ones for better performance on low-spec devices.

Ensure you have Python 3.8+ and a stable environment.

Phi-3-mini uses optional quantization (BitsAndBytesConfig) for memory efficiency.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🙋‍♂️ Author
Vimarsh Dwivedi



Built with ❤️ to make document understanding faster and easier.
