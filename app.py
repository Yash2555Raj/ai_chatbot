import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Gemini API Configuration
API_KEY = "AIzaSyCwhYEbGP40CkJMCLM73iLRUG2c1xDRsEk"  # Replace with your actual key
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Gemini API Configuration Error: {e}")
    model = None

# PDF Processing Functions
def translate_text_deep(text, target_lang="en"):
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        print("Translation error:", e)
        return text

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text().strip()
            if len(page_text) < 10:  # If text is too short, assume it's a scanned PDF
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                ocr_text = pytesseract.image_to_string(image, lang='hin+eng')
                full_text += "\n" + ocr_text
            else:
                full_text += "\n" + page_text
    return full_text.strip()

def extract_text_from_pdfs(pdf_paths):
    texts = []
    for path in pdf_paths:
        texts.append(extract_text_from_pdf(path))
    return "\n".join(texts).strip()

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def setup_vector_store(chunks, persist_dir="./chroma_db", collection="pdf_info"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )
    vector_store.add_texts(texts=chunks)
    return vector_store

def is_hindi(text):
    return any('\u0900' <= char <= '\u097F' for char in text)

# Setup PDF Processing
pdf_files = ['shivaji.pdf','Chhatrapati.pdf','ihks101.pdf']  # Update with your PDF file paths
pdf_text = extract_text_from_pdfs(pdf_files)
chunks = split_text_into_chunks(pdf_text)
vector_store = setup_vector_store(chunks)

# Chat with Gemini and Vector Store
def chat_with_gemini(prompt, context=""):
    if not model:
        return "Error: Gemini API not configured properly."
    try:
        full_prompt = f"{context}\n\nUser Query: {prompt}"
        response = model.generate_content(full_prompt)
        if response and response.text:
            return response.text.strip()
        else:
            print("Empty or invalid response from Gemini API.")
            return "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "Sorry, something went wrong. Please try again later."

def get_response(user_query):
    # Translate if Hindi
    query_en = translate_text_deep(user_query, target_lang="en") if is_hindi(user_query) else user_query

    # Retrieve relevant chunks from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    relevant_chunks = retriever.get_relevant_documents(query_en)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks]) if relevant_chunks else ""

    # Check for greetings
    greetings = ["hello", "hi", "hey", "howdy", "greetings"]
    if any(greeting in user_query.lower() for greeting in greetings):
        return "Hello! How can I assist you today?"

    # Get response from Gemini with context
    answer_en = chat_with_gemini(query_en, context=context)

    # Fallback for unclear responses
    if not answer_en or "I don't know" in answer_en.lower():
        answer_en = "I'm here to help! Could you please provide more details or ask something else?"

    # Translate back to Hindi if needed
    return translate_text_deep(answer_en, target_lang="hi") if is_hindi(user_query) else answer_en

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."})
    response_text = get_response(user_input)
    return jsonify({"response": response_text})

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files["pdf"]
    pdf_path = f"temp_{pdf_file.filename}"
    pdf_file.save(pdf_path)

    try:
        # Update global PDF processing
        global pdf_text, chunks, vector_store
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(pdf_text)
        vector_store = setup_vector_store(chunks)
        return jsonify({"message": "PDF uploaded and processed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)