import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

texts = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",    
    "Campeonato de futebol",
    "Política internacional",
    "Novo smartphone da Samsung",
    "Análise do desempenho do time",
    "Eleições municipais",
    "Lançamento de um novo filme",
    "Tecnologia de ponta",
    "Futebol feminino",
    "Eleições estaduais",
    "Novo recurso do Instagram",
    "Análise do desempenho do jogador",
    "Eleições para o Senado",
    "Novo aplicativo de mensagens"
    "Volleybol como esporte olímpico",
    "Serie de terror",
    "Novo jogo de videogame"
]

stop_words = set(stopwords.words('portuguese'))
texts = [[word for word in document.lower().split() if word not in stop_words] for document in texts]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")