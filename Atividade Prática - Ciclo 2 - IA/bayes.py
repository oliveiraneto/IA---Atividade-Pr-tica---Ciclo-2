# Aluno: Sebastião Oliveira Silva Neto - 2011478

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Carregar os dados em um dataframe
dados = pd.read_csv("dengue.csv", sep=";")

# Imprimir valores únicos de todas as colunas
for column in dados.columns:
    valores_unicos = dados[column].unique()
    print(f"Valores únicos da coluna '{column}':")
    print(valores_unicos)
    print()

# Remover pacientes sem informação sobre dengue
dados = dados[dados['tem_dengue'].notna()]

# Remover as colunas 'tipo_dengue' e 'nome_paciente'
dados.drop(columns=['tipo_dengue', 'nome_paciente'], inplace=True)

# Codificar variáveis categóricas em números
le = LabelEncoder()
dados_encoded = dados.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

# Separar os dados em características (X) e rótulos (y)
X = dados_encoded.drop(columns=['tem_dengue'])
y = dados_encoded['tem_dengue']

# Treinar o classificador Naive Bayes
clf = MultinomialNB()
clf.fit(X, y)

# Prever a ocorrência da dengue para todos os exemplos no conjunto de dados
probabilidades = clf.predict_proba(X)[:, 1]  # Probabilidade de classe positiva (ocorrência da dengue)

# Adicionar as probabilidades de ocorrência da dengue ao dataframe
dados['probabilidade_dengue'] = probabilidades

# Calcular a probabilidade média de ocorrência da dengue para cada bairro
probabilidade_media_por_bairro = dados.groupby('bairro_moradia')['probabilidade_dengue'].mean()

# Classificar os bairros com base na probabilidade média de ocorrência da dengue
bairros_ordenados = probabilidade_media_por_bairro.sort_values(ascending=False)

# Imprimir a lista ordenada dos bairros que mais devem receber atenção e ações para prevenção da dengue
print("Lista ordenada dos bairros que mais devem receber atenção e ações para prevenção da dengue:")
print(bairros_ordenados)
