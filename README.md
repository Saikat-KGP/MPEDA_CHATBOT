Our chatbot  utilizes Langchain for question answering and is used for 3 different purposes:

Queries related to seafood and seafood business
Providing description from Images(BARD)
Reading from personal csv files We used beautiful soup for scrapping data from MPEDA and other websites and cleaned the text and made a pdf. Langchain is used to do question answering from this pdf Web scraping data from the MPEDA official website and multiple seafood export company websites, followed by text cleaning, preprocessing using NLTK & re library.
EfficientNet for classifying image,Flask as the main python framework for integration,News API for showing recent news related to seafood business,Google Firebase - creating database of login,smtplib for providing follow up emails with proper summary
Instructions

git clone https://github.com/Saikat-KGP/MPEDA_CHATBOT.git
pip install bardapi
go to bard-> console-> application -> cookies-> bard link -> _Secure - 1PSID api key 4)pip install langchain
pip install openai
pip install pyPDF2
pip install faiss-cpu
pip install openai langchain sentence_transformers
pip install chromadb
pip install cohere tiktoken
pip install llmx==0.0.15a0
pip install unstructured
pip install kaleido

pip install tiktoken If tensorflow and opencv not installed in system pip install tensorflow pip install cv2 pip install opencv-python
To access BARD API KEY: os.environ['_BARD_API_KEY'] = 'your_api_key' Process to generate this API KEY: Bard website -> console -> application -> cookies -> https://bard.google.com ->_Secure - 1PSID API key

Running the application in terminal: python app.py

Work Flow 1)Signup as new user 2)MPEDA Specific Query button on Sidebar 3)News cards showing latest news 4)Enter Your required query 5)Upload your Required Image to have details 6)Upload your required CSV file to question 7)Use Translate dropdown 8)Click the Required Buttons as per your options
