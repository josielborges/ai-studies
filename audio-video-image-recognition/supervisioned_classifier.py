from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

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

labels = [
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "política",
    "multimidia"
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "política",
    "tecnologia",
    "esportes",
    "multimidia",
    "multimidia"
]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
predicted_labels = model.predict(X_test)

print(metrics.classification_report(y_test, predicted_labels, zero_division=0))

new_texts = [
    "Eleições presidenciais",
    "Campeonato de futebol",
    "Tecnologia de ponta"]

new_labels_predicted = model.predict(new_texts)

print("Novos textos e suas previsões:")
for text, label in zip(new_texts, new_labels_predicted):
    print(f"Texto: {text} - Previsão: {label}")