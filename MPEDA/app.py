from flask import Flask, render_template, request, jsonify,json, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import firebase_admin
from firebase_admin import credentials, initialize_app, auth
from werkzeug.security import generate_password_hash, check_password_hash
from bardapi import Bard
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from googletrans import Translator
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions, EfficientNetB0
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import io
from PIL import Image
import pandas as pd
from collections.abc import MutableMapping

# Your code using MutableMapping

# Some code using MutableMapping


app = Flask(__name__)

cred = credentials.Certificate('mpeda.json')
firebase_admin.initialize_app(cred)
# db = firestore.client()

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email']
    password = request.form['password']

    # Query user data from Firestore
    try:
        # Create a new user with email and password
        user = auth.create_user(
            email=email,
            password=password
        )

        return redirect(url_for('welcome'))
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    try:
        # Sign in with email and password
        user = auth.sign_in_with_email_and_password(email, password)

        # Get the user's ID token
        id_token = user['idToken']

        # Verify the ID token to authenticate the user
        decoded_token = auth.verify_id_token(id_token)

        # Access user information from decoded token
        user_id = decoded_token['uid']
        email = decoded_token['email']

        # Redirect to the welcome page or do other actions

        return redirect(url_for('welcome'))
    except ValueError as e:
        error_message = str(e)
        return f"Login failed. {error_message}"
        

@app.route('/')
def home():
    return render_template('enter.html')

@app.route('/welcome')
def welcome():
    return render_template('index.html')

chat_history = []
chat_history_string = '\n\n'.join([f"Question: {entry['question']}\nAnswer: {entry['answer']}" for entry in chat_history])
print(chat_history_string)

Summaries = []
summary = "This is the overall summery of your chatting "
summary = summary + "\n"
for i in range(len(Summaries)):
    print(type(Summaries[i]))

    if Summaries[i] is not None:
        summary = summary + str(Summaries[i])

fish_classes = [
    "coho", "lobster", "goldfish", "shark", "puffer",
    "lionfish",
    "anemone fish",
    "eel",
    "garfish",
    "grey whale",
    "killer whale",
    "king crab",
    "Alaska crab",
    "fiddler crab",
    "rock crab",
    "Dungeness crab",
    "jellyfish",
    "goldfish",
    "stingray",
    "crampfish",
    "great white shark",
    "tiger shark",
    "electric ray", "crampfish", "numbfish",
    "hammerhead", "hammerhead shark",
    "salmon", "tuna", "swordfish", "marlin", "halibut",
    "anchovy", "mackerel", "sardine", "herring", "cod",
    "catfish", "barracuda", "snapper", "grouper", "flounder",
    "squid", "octopus", "cuttlefish", "shrimp", "prawn",
    "crab", "lobster", "crawfish", "clam", "oyster",
    "mussel", "scallop", "abalone", "sea urchin", "sea cucumber"
]

def clean_paragraphs(answer):
    paragraphs = answer.split('\n\n')

    cleaned_paragraphs = []

    for paragraph in paragraphs:
        # Remove extra whitespaces
        cleaned_paragraph = re.sub(r'\s+', ' ', paragraph).strip()

        # Convert bullet points to a list
        cleaned_paragraph = re.sub(r'([*])\s+', r'\n\1 ', cleaned_paragraph)

        cleaned_paragraphs.append(cleaned_paragraph)

    return cleaned_paragraphs

def clean_string(input_string):
    # Remove HTML tags
    clean_string = re.sub('<.*?>', '', input_string)

    # Remove extra whitespaces
    clean_string = ' '.join(clean_string.split())

    # Remove newline characters
    clean_string = clean_string.replace('\n', ' ')

    # Remove ### and ** tags
    clean_string = clean_string.replace('#', '').replace('*', '')

    # Remove any additional unwanted characters or patterns
    clean_string = re.sub(r'\bbla\b', '', clean_string)

    return clean_string



model = EfficientNetB0(weights='imagenet')
from googletrans import Translator

@app.route('/upload_image', methods=['POST'])
def upload_image():
    img_path=request.files['image_upload']
    img_bytes = io.BytesIO(img_path.read())
    img = Image.open(img_bytes)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    className = decoded_predictions[0][1]
    display_text = f"give an elaborated explanation of all type of  {className} "
    print(Bard().get_answer(display_text)['content'])
    displaying_text=Bard().get_answer(display_text)['content']
    if isinstance(displaying_text, (str, bytes)):
        if isinstance(displaying_text, bytes):
            displaying_text = displaying_text.decode('utf-8')

        cleaned_paragraphs = clean_paragraphs(displaying_text)
        cleaned_text = "\n".join(cleaned_paragraphs)
        cleaned_text = clean_string(cleaned_text)
        

        for i, paragraph in enumerate(cleaned_paragraphs, start=1):
            print(f"\n{paragraph}\n")
    else:
        print("Answer is not a valid string or bytes-like object.")
    target_language = request.form['target_language']
    translation = f"answer {cleaned_text} into language code {target_language} "
    translation_answer=Bard().get_answer(translation)['content']
    translation_answer=clean_seafood_text(translation_answer)
    # translator = Translator()
    # translation = translator.translate(displaying_text, dest=target_language)
    summary = f"Write a summary within 40 words about {cleaned_text}"
    Summaries.append(Bard().get_answer(summary)['content'])
    return render_template('upload_image.html',cleaned_text=cleaned_text,translation_answer=translation_answer)

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    csv_path=request.files['csv_upload']
    df=pd.read_csv(csv_path)
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(df)
    docSearch = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm = OpenAI(), chain_type = "stuff", vectorstore = docSearch)
    prompt = f"write all insights about this dataframe {df}"
    csv_display=qa.run(prompt)
    if isinstance(csv_display, (str, bytes)):
        if isinstance(csv_display, bytes):
            csv_display = csv_display.decode('utf-8')

        cleaned_paragraphs = clean_paragraphs(csv_display)
        cleaned_text1 = "\n".join(cleaned_paragraphs)
        cleaned_text1 = clean_string(cleaned_text1)

        for i, paragraph in enumerate(cleaned_paragraphs, start=1):
            print(f"\n{paragraph}\n")
    else:
        print("Answer is not a valid string or bytes-like object.")

    target_language = request.form['target_language']
    translation = f"answer {cleaned_text1} into language code {target_language} "
    translation_answer=Bard().get_answer(translation)['content']
    translation_answer=clean_seafood_text(translation_answer)

    summary = f"Write a summary within 40 words about {cleaned_text1}"
    Summaries.append(Bard().get_answer(summary)['content'])
    return render_template('upload_csv.html',cleaned_text1=cleaned_text1,translation_answer=translation_answer)



os.environ["OPENAI_API_KEY"] = "PUT_YOUR_OWN_OPENAI_API_KEY"
os.environ["SERPAPI_API_KEY"] = "PUT_YOUR_OWN_SERP_API_KEY"
chain = load_qa_chain(OpenAI(), chain_type="stuff")


pdfreader = PdfReader('important.pdf')
from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)





def clean_seafood_text(seafood_text):
    cleaned_text = re.sub(r'\*{2,}', '', seafood_text)
    cleaned_text = re.sub(r'\[Image of [^\]]*\]', '', cleaned_text)

    cleaned_text = re.sub(r'\bbla\b.*?(\.|$)', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = cleaned_text.replace('\n', '')
    return cleaned_text.strip()

os.environ['_BARD_API_KEY'] = 'PUT_YOUR_OWN_BARD_API_KEY'

chat_history1 = [
    {'question': 'What is your name?', 'answer': 'My name is ChatGPT.'},
    {'question': 'How does this work?', 'answer': 'I generate responses based on input.'},
]
chat_history_string1 = '\n\n'.join([f"Question: {entry['question']}\nAnswer: {entry['answer']}" for entry in chat_history1])

print(chat_history_string1)
print(type(chat_history_string1))

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdfreader = PdfReader('MPEDA PROMPTS.pdf')

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}

def load_single_pdf(pdf_filepath):
    with open(pdf_filepath, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()
    return text

def split_docs(document, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Create a Document object with the text content
    document_obj = Document(document)

    # Use the Document object in the splitter
    docs = text_splitter.split_documents([document_obj])
    return docs

# Example usage
pdf_filepath = 'MPEDA PROMPTS.pdf'  # Replace with the actual path to your PDF file
document_text = load_single_pdf(pdf_filepath)

# Assuming you want to split each document into chunks of 1000 characters with a 20-character overlap
chunk_size = 1000
chunk_overlap = 20
chunked_docs = split_docs(document_text, chunk_size, chunk_overlap)

# Print the number of resulting chunks
print(len(chunked_docs))

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.vectorstores import Chroma
db = Chroma.from_documents(chunked_docs, embeddings)
query = "Which ornamental fish can India provide?"
matching_docs = db.similarity_search(query)
matching_docs = db.similarity_search_with_score(query,k=2)
matching_docs

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=chunked_docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()
new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

from langchain.chains.question_answering import load_qa_chain
# chain = load_qa_chain(llm, chain_type="stuff")
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    input_text = request.form['input_text']
    target_language = request.form['target_language']
    translator = Translator()
    db = Chroma.from_documents(chunked_docs, embeddings)
    matching_docs = db.similarity_search_with_score(query,k=2)

    answer =  chain.run(input_documents=matching_docs, question=input_text)
    original=clean_seafood_text(answer)
    
    translation = f"answer {input_text} into language code {target_language} "
    translation_answer=Bard().get_answer(translation)['content']
    translation_answer=clean_seafood_text(translation_answer)
    

    chat_history.append({'question':input_text, 'answer': original,'translation':translation_answer})
    summary = f"Write a summary within 40 words about {answer}"
    Summaries.append(Bard().get_answer(summary)['content'])

    return render_template('after.html', chat_history=chat_history,input_text=input_text, translation_answer=translation_answer)



@app.route('/mpeda_specific')
@app.route('/mpeda_specific.html')
def mpeda_specific():
    return render_template('mpeda_specific.html')

result_string = ', '.join(Summaries)
print(type(result_string))

@app.route('/send_email', methods=['GET'])
def send_email():
    sender = 'srinjoydas566@gmail.com'
    password = 'pxer uuxr uylz xxwg'
    receiver = 'saikat02004@gmail.com'
    subject = 'TEST for mpeda'
    

    em = EmailMessage()
    em['From'] = sender
    em['To'] = receiver
    em['Subject'] = subject
    em.set_content(summary)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', port=465, context=context) as smtp:
        smtp.login(sender, password)
        smtp.sendmail(sender, receiver, em.as_string())


@app.route('/mpeda_specific_after', methods=['POST'])
def mpeda_specific_after():
    input_parameter = request.form['input_parameter']
    docs = document_search.similarity_search(input_parameter)
    uttor=chain.run(input_documents=docs, question=input_parameter)
    return render_template('mpeda_specific_after.html',uttor=uttor)
    

if __name__ == '__main__':
    app.run(debug=True)
