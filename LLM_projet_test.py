import streamlit as st
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
from PIL import Image
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pytesseract

# Charger la clé API OpenAI depuis le fichier .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Configuration des pages Streamlit
st.set_page_config(
    page_title="Application AI : Traduction et Analyse avec LangChain",
    page_icon="📚",
    layout="wide",
)

# Sidebar pour navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à",
    ["Accueil", "Traduction", "Résumé PDF", "Questions sur le PDF", "Analyse des Images", "Analyse avec LangChain"]
)


# Fonction pour diviser le texte pour la traduction
def split_text_for_translation(text, max_tokens=3000):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Fonction pour traduire un texte
def translate_text(text, target_language="fr"):
    chunks = split_text_for_translation(text, max_tokens=3000)
    translations = []

    for i, chunk in enumerate(chunks):
        prompt = f"Translate the following text to {target_language}:\n\n{chunk}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in translation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            translations.append(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            translations.append(f"Erreur lors de la traduction du morceau {i + 1}: {str(e)}")

    # Combiner toutes les traductions
    combined_translation = " ".join(translations)
    return combined_translation


# Fonction pour extraire du texte d'un fichier PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Fonction pour diviser le texte pour d'autres usages
def split_pdf_text_into_chunks(text, max_tokens=3000):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Fonction pour générer un résumé
def summarize_text(text):
    chunks = split_pdf_text_into_chunks(text, max_tokens=2000)  # Divisez le texte en morceaux
    summaries = []
    
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize the following text:\n\n{chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in summarizing text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        summaries.append(response['choices'][0]['message']['content'].strip())
    
    combined_summary = " ".join(summaries)
    return combined_summary


# Fonction LangChain : Charger un PDF et poser une question
def load_pdf_and_split(file_path, chunk_size=3000, chunk_overlap=200):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def ask_question_with_langchain(file_path, question):
    docs = load_pdf_and_split(file_path)

    llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

    chain = load_qa_chain(llm, chain_type="stuff")

    responses = []
    for i, doc in enumerate(docs):
        try:
            response = chain.run(input_documents=[doc], question=question)
            responses.append(f"Morceau {i + 1}: {response}")
        except Exception as e:
            responses.append(f"Morceau {i + 1}: Erreur - {str(e)}")

    final_response = "\n\n".join(responses)
    return final_response


# Fonction pour poser une question sur un PDF
def ask_question_on_chunks(text, question):
    chunks = split_pdf_text_into_chunks(text, max_tokens=3000)
    responses = []

    for i, chunk in enumerate(chunks):
        prompt = f"Here is a portion of the text: {chunk}\n\nQuestion: {question}\nAnswer:"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            responses.append(f"Chunk {i + 1}: {response['choices'][0]['message']['content'].strip()}")
        except Exception as e:
            responses.append(f"Chunk {i + 1}: Erreur lors du traitement - {str(e)}")

    final_answer = "\n\n".join(responses)
    return final_answer


# Page : Accueil
if page == "Accueil":
    st.title("Bienvenue dans l'Application AI 🚀")
    st.write("""
    Cette application combine les fonctionnalités OpenAI et LangChain pour :
    - Traduire des textes.
    - Générer des résumés de documents PDF.
    - Poser des questions sur des documents.
    - Visualiser et annoter des images.
    """)
    image_path = os.path.expanduser("~/Documents/DOSSIER LLM Large language modèle/ugur-akdemir-XT-o5O458as-unsplash.jpg")
    image = Image.open(image_path)
    st.image(image, caption="Bienvenue dans l'Application AI", use_container_width=True)


# Page : Traduction
elif page == "Traduction":
    st.title("Traduction de Texte 🌐")
    text_to_translate = st.text_area("Entrez un texte à traduire", height=200)
    target_language = st.selectbox("Choisissez une langue cible", ["fr", "en", "es", "de", "it"])
    if st.button("Traduire"):
        if text_to_translate:
            try:
                translated_text = translate_text(text_to_translate, target_language)
                st.text_area("Texte traduit", translated_text, height=200)
            except Exception as e:
                st.error(f"Erreur lors de la traduction : {str(e)}")
        else:
            st.warning("Veuillez entrer un texte à traduire.")


# Page : Résumé PDF
elif page == "Résumé PDF":
    st.title("Générer un Résumé 📝")
    uploaded_file = st.file_uploader("Chargez un fichier PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Contenu du PDF", pdf_text, height=300)
        if st.button("Générer un résumé"):
            summary = summarize_text(pdf_text)
            st.text_area("Résumé du PDF", summary, height=200)


# Page : Questions sur le PDF
elif page == "Questions sur le PDF":
    st.title("Questions sur le contenu du PDF 🤔")
    uploaded_file = st.file_uploader("Chargez un fichier PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Contenu du PDF", pdf_text, height=300)
        question = st.text_input("Posez une question sur le contenu du PDF")
        if st.button("Poser la question"):
            if question:
                try:
                    answer = ask_question_on_chunks(pdf_text, question)
                    st.text_area("Réponse générée", answer, height=300)
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
            else:
                st.warning("Veuillez entrer une question avant de cliquer sur 'Poser la question'.")


# Page : Analyse avec LangChain
elif page == "Analyse avec LangChain":
    st.title("Analyse avancée avec LangChain 📘")
    st.write("Chargez un fichier PDF et posez une question pour utiliser LangChain.")
    uploaded_file = st.file_uploader("Chargez un fichier PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        question = st.text_input("Posez une question sur le PDF")
        if st.button("Analyser avec LangChain"):
            if question:
                try:
                    response = ask_question_with_langchain("temp.pdf", question)
                    st.text_area("Réponses générées", response, height=300)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse avec LangChain : {str(e)}")
            else:
                st.warning("Veuillez entrer une question avant de cliquer sur 'Analyser avec LangChain'.")


# Page : Analyse des Images
elif page == "Analyse des Images":
    st.title("Analyse et Annotation d'Images 🖼️")
    uploaded_image = st.file_uploader("Chargez une image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Image chargée", use_container_width=True)

        # Résumé via OpenAI
        if st.button("Générer un résumé de l'image"):
            try:
                prompt = "Describe this image in detail."
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in image descriptions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150
                )
                st.success(f"Résumé généré : {response['choices'][0]['message']['content']}")
            except Exception as e:
                st.error(f"Erreur lors de la génération du résumé : {str(e)}")

        # Analyse des couleurs dominantes
        if st.button("Analyser les couleurs dominantes"):
            img_array = np.array(image)
            img_array = img_array.reshape((-1, 3))
            n_colors = 5
            kmeans = KMeans(n_clusters=n_colors)
            kmeans.fit(img_array)
            colors = kmeans.cluster_centers_
            fig, ax = plt.subplots()
            ax.pie([1] * n_colors, colors=colors / 255, labels=[f"Couleur {i + 1}" for i in range(n_colors)])
            ax.set_title("Couleurs dominantes")
            st.pyplot(fig)

        # OCR
        if st.button("Extraire le texte de l'image (OCR)"):
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    st.text_area("Texte extrait", text, height=200)
                else:
                    st.warning("Aucun texte détecté dans l'image.")
            except Exception as e:
                st.error(f"Erreur lors de l'OCR : {str(e)}")
