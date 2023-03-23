# Criação de um modelo preditivo que, com base nos dados de treinamento, treine e 
# entenda as correlações de todos os recursos e depois aplicar o modelo em diferentes clientes 
# e elaborar uma predição de probabilidade do cliente sair do banco 

## Pré processamento dos dados 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 

def get_started():
    
    # # # # PRE PROCESSAMENTO
    df = pd.read_csv('files/training_data.csv') 
    
    # Lista de dados com as caracteristicas
    x = df.iloc[:, 3:-1].values
    # Dados em lista da variavel dependente, a variavel que queremos prever
    y = df.iloc[:, -1].values
    
    # Transformar a variavel genero em dado categorico -> Gênero -> 0 ou 1 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x[:, 2] = le.fit_transform(x[:, 2]) ### Máquina gerou automaticamente que female é 0 e male é 1 
    
    # Transformar variável não categorica em variável numérica -> 'Geography' column
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') #[1] é o index da coluna
    x = np.array(ct.fit_transform(x))
    
    print(x)
    
    # Partir o dataset em treino e teste 
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Feature scaling: obrigatório para deep learning! 
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    # Começar a ANN 
    ann = tf.keras.models.Sequential()
    
    #Input hidden layer 
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    # Output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    # # # # Compilação ANN
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(ann.fit(x_train, y_train, batch_size=32, epochs=100))
    print('-'*30)
    print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
    


if __name__ == '__main__':
    get_started()