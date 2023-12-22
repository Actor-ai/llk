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
from transformers import pipeline


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
llm_model_name = os.environ['LLM']
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

queries_mk_r1 = [
    'Государственная услуга "Выдача прокатных удостоверений на фильмы, созданные в Российской Федерации или приобретенные за рубежом для проката на ее территории, и ведение Государственного регистра фильмов" (осуществляется через ЕПГУ)',
    'Доля обращений заявителей для получения государственной услуги в электронном виде от общего количества обращений',
    'Доля результатов предоставления государственной услуги заявителю исключительно в электронном виде от общего количества результатов',
    'Регламентное время предоставления государственной услуги',
    'Подача заявления без личного посещения ведомства',
    'Проактивное предоставление услуги',
    'Экстерриториальный принцип предоставления государственной услуги',
    'Автоматическое принятие решения без участия человека при предоставлении государственной услуги',
    'Результат государственной услуги в электронном виде является электронным юридически значимым документом',
    'Результат предоставления государственной услуги заносится в реестр юридически значимых записей',
    'Уровень удовлетворенности граждан качеством оказания государственных услуг',
    'Возможность обращения заявителей на ЕПГУ для получения государственной услуги',
    'Результат предоставления государственной услуги заявителю в электронном виде на ЕПГУ',
    'Государственная услуга "Выдача удостоверений национального фильма" (осуществляется через ЕПГУ)',
    'Оценка качества оказания социально ориентированной некоммерческой организацией общественно полезных услуг',
    'Государственная услуга "Выдача разрешительных документов на вывоз, временный вывоз культурных ценностей из Российской Федерации" (осуществляется через ЕПГУ)',
    'Выдача разрешения и задания на проведение работ по сохранению объекта культурного наследия (памятника истории и культуры) народов Российской Федерации федерального значения',
    'Государственная услуга "Лицензирование деятельности по сохранению объектов культурного наследия (памятников истории и культуры) народов Российской Федерации" (осуществляется через ЕПГУ)',
    'Государственная услуга "Согласование проектной документации на проведение работ по сохранению объекта культурного наследия (памятника истории и культуры) народов Российской Федерации федерального значения" (осуществляется через ЕПГУ)',
    'Государственная услуга "Регистрация фактов пропажи, утраты, хищения культурных ценностей, организация и обеспечение оповещения государственных органов и общественности в Российской Федерации и за ее пределами об этих фактах" (осуществляется через ЕПГУ)',
    'Государственная услуга "Принятие решения о включении музейных предметов и музейных коллекций в состав негосударственной части Музейного фонда Российской Федерации, исключение из состава Музейного фонда Российской Федерации" (осуществляется через ЕПГУ)',
    'Государственная услуга "Выдача разрешений (открытых листов) на проведение работ по выявлению и изучению объектов археологического наследия" (осуществляется через ЕПГУ)',
    '"Предоставление социальной поддержки молодежи от 14 до 22 лет для повышения доступности организаций культуры"',
    'Аттестация экспертов по проведению государственной историко-культурной экспертизы',
    'Выдача паспорта на струнные смычковые музыкальные инструменты или смычки, отнесенные по результатам экспертизы культурных ценностей к культурным ценностям, имеющим особое значение, или не отнесенные к культурным ценностям либо к культурным ценностям, в отношении которых правом Евразийского экономического союза установлен разрешительный порядок вывоза',
    'Аттестация лиц на право проведения реставрационных работ в отношении музейных предметов и музейных коллекций',
    'Аттестация специалистов в области сохранения объектов культурного наследия (за исключением спасательных археологических полевых работ), в области реставрации иных культурных ценностей',
    'Аттестация экспертов по культурным ценностям',
    'Выдача специального разрешения на передачу музейных предметов и музейных коллекций, включенных в состав Музейного фонда Российской Федерации, от одного лица другому при отчуждении, универсальном правопреемстве и в иных случаях',
    'Государственный контроль и надзор за состоянием Музейного фонда Российской Федерации, за деятельностью негосударственных музеев в Российской Федерации',
    'Доля проверок, проведенных дистанционно в электронном виде, от общего количества проведенных проверок',
    'Реестр объектов, отнесенных к категориям рисков/классам опасности на основании модели рисков',
    'Межведомственное взаимодействие при осуществлении контрольных (надзорных) мероприятий в электронном виде',
    'Юридически значимые уведомления и документооборот с контролируемыми лицами',
    'Принятие решений на основании данных инструментальных средств мониторинга и контроля',
    'Обязательные требования по виду контроля (надзора) систематизированы в ФГИС «Реестр обязательных требований» и используется в проверочных листах',
    'Обеспечено обжалование решений органа контроля (надзора), действий/бездействия должностных лиц полностью в электронном виде',
    'Оценка эффективности и результативности инспекторов происходит на основе утвержденной системы показателей',
    'Государственный контроль за соблюдением особого режима хранения и использования национального библиотечного фонда',
    'Государственный контроль и надзор за соблюдением законодательства Российской Федерации в отношении культурных ценностей, перемещенных в Союз ССР в результате Второй мировой войны и находящихся на территории Российской Федерации, а также за сохранностью перемещенных культурных ценностей и их учетом',
    'Федеральный государственный надзор за состоянием, содержанием, сохранением, использованием, популяризацией и государственной охраной отдельных объектов культурного наследия федерального значения, перечень которых устанавливается Правительством Российской Федерации',
    'Сбор статистической отчетности (АИС Статистика)',
    'Доля показателей в АИС Статистика, по которым ведется автоматический мониторинг качества данных в части их полноты, корректности, актуальности и связанности с использованием ФЛК (форматно-логический контроль)',
    'Доля учреждений культуры, использующих ЭЦП для подписания данных статистических форм',
    'Реализация государственной функции осуществляется через ЕПГУ',
    'Управление государственной политикой (ЕАС)',
    'Доля регионов РФ, использующих в своей деятельности ЕАС (Единая аналитическая система) на постоянной основе',
    'Сохранение музейного фонда (Госкаталог)',
    'Количество музеев, зарегистрированных в реестре музеев',
    'Доля музеев осуществляющих работу в реестре сделок',
    'Количество музеев, осуществляющих авторизацию через ЕСИА',
    'Процент сделок подписанных ЭЦП от общего количества сделок оформленных в Госкаталоге',
    'Доля изображений предметов, зарегистрированных в Государственном каталоге Музейного фонда РФ, модерация которых осуществлялась с использованием механизмов машинного зрения',
    'Популяризация культурного наследия (Pro.culture.ru)',
    'Доля мест от общего числа учреждений культуры',
    'Доля активных учреждений культуры',
    'Количество информационных ресурсов, на которых установлен счетчик Цифровая Культура',
    'Общее кол-во событий (мероприятий), о которых информировано население',
    'Доля анонсов мероприятий в сфере культуры на платформе PRO.Культура.РФ модерация которых происходит с использованием с использованием механизмов анализа естественного языка и машинного зрения.',
    'Популяризация культурного наследия (Культура.РФ)',
    'Посещаемость (визиты)',
    'Глубина просмотра',
    'Время на сайте',
    'Количество контентных единиц',
    'Повышение уровня надежности и безопасности информационных систем, технологической независимости информационно-технологической инфраструктуры от оборудования и программного обеспечения, происходящих из иностранных государств',
    'Доля расходов на закупки и/или аренду отечественного программного обеспечения и платформ от общих расходов на закупку или аренду программного обеспечения',
    'Доля сотрудников, подключенных к системе электронного документооборота',
    'Доля расходов на закупки и/или аренду радиоэлектронной продукции (в том числе систем хранения данных и серверного оборудования, автоматизированных рабочих мест, программно-аппаратных комплексов, коммуникационного оборудования, систем видеонаблюдения) российского происхождения от общих расходов на закупку или аренду радиоэлектронной продукции',
    'Доля отечественного программного обеспечения и компонентов, используемых в государственных информационных системах',
    'Доля программного обеспечения, установленного и используемого на рабочих местах в ведомстве, не включенного в единый реестр российских программ для электронных вычислительных машин и баз данных и национальный фонд алгоритмов и программ для электронных вычислительных машин',
    'Доля используемых отечественных средств защиты информации',
    'Доля используемых средств защиты информации, странами происхождения которых являются иностранные государства, совершающие в отношении Российской Федерации, российских юридических лиц и физических лиц недружественные действия, либо производителями которых являются организации, находящиеся под юрисдикцией таких иностранных государств, прямо или косвенно подконтрольные им либо аффилированные с ними',
    'Обеспечена готовность перевода компонентов государственных информационных систем на единую цифровую платформу Российской Федерации «Гостех» в соответствии с утвержденными планами',
    'Обеспечена архитектурная целостность ИС (единый ландшафт)',
    'Проведен аудит ГИС',
    'Обеспечение функционирования информационных систем и компонентов информационно-телекоммуникационной системы',
    'Доступность информационных систем класса защищенности К1, установленного в соответствии с Требованиями о защите информации',
    'Доступность информационных систем класса защищенности К2 и менее, установленного в соответствии с Требованиями о защите информации, и иных информационных систем',
    'Доля информационных систем, имеющих действующий аттестат соответствия требованиям информационной безопасности',
    'Доля государственных информационных ресурсов доступных в режиме онлайн через витрины данных посредством СМЭВ',
    'Обеспечено взаимодействие с Национальным координационным центром по компьютерным инцидентам (НКЦКИ) в рамках Государственной системы обнаружения, предупреждения и ликвидации последствий компьютерных атак на информационные ресурсы Российской Федерации (ГосСОПКА)',
    'Доля государственных информационных систем, для которых обеспечена возможность взаимодействия в электронной форме с гражданами (физическими лицами) и организациями в соответствии с правилами и принципами, установленными национальными стандартами Российской Федерации в области криптографической защиты информации',
    'Используется ФГИС "Единая система идентификации и аутентификации" ЕСИА',
    'ГИС о государственных и муниципальных платежах ГМП (ГИС ГМП)',
    'Доля государственных информационных систем Минкультуры России, переведенных в ГосОблако, от общего количества государственных информационных систем Минкультуры России',
    'Доля информационных систем, обеспечивающих ведение информационных ресурсов, описанных в ФГИС «ЕИП НСУД»',
    'Количество видов сведений, предоставляемых в режиме «онлайн» органами государственной власти в рамках межведомственного взаимодействия при предоставлении государственных услуг и исполнения функций, в том числе коммерческих организаций в соответствии с законодательством',
    'Количество внедрённых ведомственных витрин данных',
    'Количество доступных дата-сетов (наборов данных), используемых для решения задач с применением технологий искусственного интеллекта',
    'Количество наборов данных, предоставляемых в целях информационно-аналитического обеспечения деятельности и поддержки принятия управленческих решений',
    'Доля инцидентов качества данных, закрытых в срок'
]

queries_mk_r2 = [
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
st.title('Akilli')
st.header('Проверка ТЗ на соответствие требованиям ВПЦТ')

with st.spinner("initialize application..."):
    pipe_fb = pipeline("translation", model="facebook/wmt19-en-ru", max_length=1024, device=0)

option = st.selectbox(
    'Выберите ВПЦТ',
    ('МинОбрНаука', 'МинКультуры раздел 1', 'МинКультуры раздел 2'))

if option == 'МинКультуры раздел 1':
    queries = queries_mk_r1
elif option == 'МинКультуры раздел 2':
        queries = queries_mk_r2
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
 #       llm = Ollama(base_url='http://ollama_llm:11434', model="zephyr:7b-beta")
        llm = Ollama(base_url='http://ollama_llm:11434', model=llm_model_name, temperature=0)


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
            Answer "yes" or "no". Think step by step. Give a short answer.
        """

        st.subheader("Требование")
        st.write(query)
        st.subheader("Анализ")
        # st.write(pipe_fb(llm(prompt))[0]['translation_text'])
        st.write(llm(prompt))


    del model, df, docs, result, splitted_text, documents, uploaded_file
    gc.collect()


