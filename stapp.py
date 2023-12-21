import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import numpy
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import gc
import random
import filetype
from langchain.llms import Ollama


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)
register_adapter(numpy.float32, addapt_numpy_float32)
register_adapter(numpy.int32, addapt_numpy_int32)

# embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/ai-forever/sbert_large_mt_nlu_ru")
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model_path = 'all-MiniLM-L6-v2'
# model_path = 'ai-forever/sbert_large_mt_nlu_ru' # threshold 0.13, dimesionality 1024
# model = SentenceTransformer(model_path)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=256,
    length_function=len,
    is_separator_regex=False,
)


def setup_database(table_name, dimensions):
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                DROP TABLE IF EXISTS llk_embeddings CASCADE;
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {table_name} (
                    text_split TEXT UNIQUE,
                    ltext_split TEXT UNIQUE,
                    embedding VECTOR({dimensions})
                );
                CREATE INDEX IF NOT EXISTS idx_embedding ON {table_name} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
            """)
            conn.commit()


def add_docs(table_name, docs):
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for doc in docs:
                text = doc.page_content
                ltext = doc.page_content.lower()
                embedding = model.encode(ltext)
#                embedding = model.embed_query(ltext)
                cur.execute(
                    f"""
                    INSERT INTO {table_name} (text_split, ltext_split, embedding) 
                    VALUES (%s, %s, %s) 
                    ON CONFLICT DO NOTHING
                    """,
                    (text, ltext, embedding.tolist())
                )
            conn.commit()
    gc.collect()
    return {"message": "Docs added successfully"}


def get_candidates(table_name, query, k):

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            embedding = model.encode(query.lower())
#            embedding = model.embed_query(query.lower())
            # K is the number of top K nearest neighbors
            cur.execute(f"""
                SELECT
                    text_split, embedding <=> '{embedding.tolist()}' AS distanse 
                    FROM {table_name} 
                    ORDER BY distanse 
                    LIMIT {k}
            """)
            top_k_candidates = cur.fetchall()
    return top_k_candidates


db_host = os.environ['APP_DB_HOST']
db_user = os.environ['APP_DB_USER']
db_password = os.environ['APP_DB_PASS']
db = os.environ['APP_DB']
models_dir = "./models"

# Database connection details
DATABASE_URL = f"postgres://{db_user}:{db_password}@{db_host}:5432/{db}"
EMBEDDING_DIM = 1024  # Adjust according to your embedding model

queries_mon = [
    "Цифровизация системы государственной научной аттестации, в том числе развитие ФИС ГНА",
    "Развитие суперсервиса \"Поступление в ВУЗ онлайн\" и создание типовых решений для образовательных организаций высшего образования, в том числе за счет развития ГИС \"Современная цифровая образовательная среда\"",
    "Внедрение и развитие на базе инфраструктуры НИКС набора специализированных сервисов в интересах сферы науки и образования России",
    "Развитие информационно-аналитической системы формирования и распределения квоты приема иностранных граждан и лиц без гражданства, в том числе соотечественников, проживающих за рубежом, на обучение в Российской Федерации (ИАС ФРКП)",
    "Создание комплексов цифровых сервисов и решений в сфере науки и высшего образования в рамках реализации домена «Наука и инновации»",
    "Развитие внутренней информационно-телекоммуникационной инфраструктуры",
    "Обеспечение функционирования ФИС ГНА",
    "Обеспечение функционирования информационно-аналитической системы формирования и распределения квоты приема иностранных граждан и лиц без гражданства, в том числе соотечественников, проживающих за рубежом, на обучение в Российской Федерации (ИАС ФРКП)",
    "Обеспечение функционирования единой государственной информационной системы учета научно-исследовательских, опытно-конструкторских и технологических работ гражданского назначения (ЕГИСУ НИОКТР)",
    "Информационно-технологическое обеспечение управления ведомственной цифровой трансформацией в сфере науки и высшего образования",
    "Обеспечение функционирования ГИС «Современная цифровая образовательная среда»",
    "Обеспечение функционирования единой цифровой платформы научного и научно-технического взаимодействия исследователей для проведения исследований",
    "Обеспечение функционирования НИКС",
    "Обеспечение функционирования информационных систем типовой деятельности",
    "Обеспечение функционирования компонентов информационно-телекоммуникационной инфраструктуры",
]

queries_mk = [
    "Цифровая трансформация государственного управления, государственных услуг (функций), контрольной надзорной деятельности",
    "Развитие платформы АИС ЕИПСК (PRO Культура)",
    "Развитие Единой автоматизированной системы поддержки оказания государственных услуг Минкультуры России:",
    "Развитие Государственного каталога музейного фонда Российской Федерации",
    "Развитие СЭД Дело",
    "Развитие АИС ЕГРОКН",
    "Создание и развитие информационно-телекоммуникационной инфраструктуры и технологических сервисов",
    "Обеспечение функционирования информационных систем и компонентов информационно-телекоммуникационной системы",
    "Единый государственный реестр памятников истории и культуры народов Российской Федерации",
    "Эксплуатация АИС Статистика",
    "Эксплуатация Государственного каталога музейного фонда Российской Федерации",
    "Эксплуатация платформы открытых данных Минкультуры России",
    "Эксплуатация ЕАС Минкультуры России",
    "Эксплуатация АИС УПБ",
    "Эксплуатация единой интеграционной платформы",
    "Эксплуатация АИС ЕИПСК (PRO Культура)",
    "Эксплуатация официального сайта Минкультуры России",
    "Эксплуатация платформы дополненной реальности",
    "Эксплуатация портала Культура.РФ",
    "Эксплуатация системы защиты информации",
    "Эксплуатация информационной системы БОР-навигатор",
    "Эксплуатация ЕАС \"Госуслуги\"",
    "Эксплуатация ИАС УПФД",
    "Эксплуатация Информационно-справочных сервисов",
    "Эксплуатация информационно-коммуникационной инфраструктуры",
    "Эксплуатация рабочих станций",
    "Эксплуатация услуг связи",
    "Эксплуатация СЭД Дело",
]


table_name = f"table_{random.randint(0, 1000000)}"

# st.set_page_config(layout="wide")
st.title('Проверка ТЗ на соответствие требованиям ВПЦТ')

option = st.selectbox(
    'Выберите ВПЦТ',
    ('МинОбрНаука', 'МинКультуры'))

if option == 'МинКультуры':
    queries = queries_mk
else:
    queries = queries_mon

uploaded_file = st.file_uploader("Choose a file. PDF and DOCX formats are supported.")
if uploaded_file is not None:
    st.text(f"File is uploaded: {uploaded_file.name}")

    file_path = os.path.join(os.getcwd(), 'uploaded', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        f.close()

    with st.spinner("Initialize embedding model..."):
        model_path = 'ai-forever/sbert_large_mt_nlu_ru'  # threshold 0.13, dimesionality 1024
        model = SentenceTransformer(model_path, cache_folder='./models_cache')
#        model = OllamaEmbeddings(model="mistral:7b")

    with st.spinner("Initialize LLM service..."):
 #       llm = Ollama(base_url='http://ollama_llm:11434', model="mistral:7b-instruct")
 #       llm = Ollama(base_url='http://ollama_llm:11434', model="mistral:7b-instruct-v0.2-q8_0")
 #       llm = Ollama(base_url='http://ollama_llm:11434', model="mistral:7b-instruct-v0.2-fp16")
        llm = Ollama(base_url='http://ollama_llm:11434', model="orca2:13b")
 #       llm = Ollama(base_url='http://ollama_llm:11434', model="zephyr:7b-beta")

    with st.spinner('Preprocess...'):

        kind = filetype.guess(file_path)
        if kind.mime == 'application/pdf':
            loader = PyPDFLoader(file_path)
        elif kind.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(file_path)
        else:
            st.text("Unsupported file format")
            st.stop()

#        loader = PyPDFLoader(file_path)
        documents = loader.load()

        pattern = r"^\d+"

        splitted_text = [re.sub(pattern, '', doc.page_content) for doc in documents]
        splitted_text = [text.replace("\n", "").strip() for text in splitted_text]
        doc_text = " ".join(splitted_text)

        docs = text_splitter.create_documents([doc_text])
        # texts = text_splitter.split_text(doc_text)

    with st.spinner('Build index...'):
        setup_database(table_name, EMBEDDING_DIM)
        add_docs(table_name, docs)

    with st.spinner('Check requirements...'):
        result = []
        for query in queries:
            candidates = get_candidates(table_name, query, 1)
            for candidat in candidates:
                result.append(
                    [
                        query, candidat[0], candidat[1]
                    ]
                )
        df = pd.DataFrame(result, columns=['Requirement', 'Context', 'Score'])

    threshold = 0.13
    df['Satisfied'] = df['Score'].apply(lambda x: True if x < threshold else False)

    st.header("Анализ ТЗ на основе близости векторов текста требования и ТЗ")

    st.text("ТЗ удовлетворяет требованиям ВПЦТ" if True in df['Satisfied'].unique() else "ТЗ НЕ удовлетворяет требованиям ВПЦТ")

    st.dataframe(df)

    st.header("Анализ ТЗ большой языковой моделью (LLM)")

    for query in queries:
        candidates = get_candidates(table_name, query, 1)

        context = candidates[0][0]

        prompt = f"""
            Look on the context and requirement:

            Context:{context}

            Requirement: {query}

            Answer the question: Does the given context satisfy the requirement?
            Answer "yes" or "no". Answer in native Russian only!
        """

        st.subheader("Требование")
        st.write(query)
        st.subheader("Анализ")
        st.write(llm(prompt))


    del model, df, docs, result, splitted_text, documents, uploaded_file
    gc.collect()


