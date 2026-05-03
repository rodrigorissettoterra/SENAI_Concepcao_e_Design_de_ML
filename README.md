# SENAI: Concepção e Design de Machine Learning
## Exercícios realizados durante o curso

### Preparação e pré-processamento de dados

#### Situação-problema1: Contextualização

<b>Dados sobre hepatite</b>

  Você assumiu o posto de cientista de dados do Hospital Santa Ajuda e lhe foi pedido que auxilie os médicos na identificação de fatores prognósticos e sobrevivência dos pacientes com hepatite. No caso, trabalhará no pré-processamento de um conjunto de dados que envolvem esses pacientes para criação de um modelo preditivo. 
  A base de dados a ser utilizada foi disponibilizada primeiramente em 1988 (GONG, 1988) no repositório da UCI e é bem estabelecida na literatura como um projeto de entrada no mundo da ciência de dados (GOH, 2020). A base apresenta 19 atributos de 155 pacientes, que incluem dados sobre o seu perfil, presença de sinais e sintomas da doença, resultados
de exames e testes em laboratório, e protocolo de tratamento adotado, além de incluir o rótulo indicando se o paciente com a doença veio a falecer. 
  Após o pré-processamento deste conjunto de dados, é esperado que eles não possuam inconsistências, redundâncias, valores faltantes, ruído e que estejam normalizados para serem utilizados como entrada por um modelo de aprendizado de máquina.
  Desse modo, você deverá superar o seguinte desafio:
• Preparar o conjunto de dados sobre hepatite.

</br>

<b> Atividade 1: Preparação inicial do conjunto de dados de hepatite </b>

  Nesta atividade, você, como cientista de dados do Hospital Santa Ajuda, trabalhará no pré-processamento de um conjunto de dados que envolvem pacientes com hepatite para criação de um modelo preditivo que auxilie os médicos na identificação de fatores prognósticos e sobrevivência dos pacientes com essa patologia.
  A base de dados a ser utilizada foi disponibilizada primeiramente em 1988 (GONG, 1988) no repositório da UCI e é bem estabelecida na literatura como um projeto de entrada no mundo da ciência de dados (GOH, 2020). Ela apresenta 19 atributos de 155 pacientes, que incluem dados sobre o seu perfil, presença de sinais e sintomas da doença, resultados de exames e testes em laboratório, e protocolo de tratamento adotado, além de incluir o rótulo indicando se o paciente com a doença veio a falecer.
  Após o pré-processamento deste conjunto de dados é esperado que eles não possuam inconsistências, redundâncias, valores faltantes, ruído e que estejam normalizados para serem utilizados como entrada por um modelo de aprendizado de máquina.
  O arquivo “.csv” da base é de livre acesso on-line e pode ser obtido através dos seguintes links, Disponível em:</br>
- <a href="https://www.openml.org/data/get_csv/55/dataset_55_hepatitis.arff" target="_blank"><u><i>https://www.openml.org/data/get_csv/55/dataset_55_hepatitis.arff</i></u></a></br>
- <a href="https://datahub.io/machine-learning/hepatitis/r/hepatitis.csv" target="_blank"><u><i>https://datahub.io/machine-learning/hepatitis/r/hepatitis.csv</i></u></a>
  Assim, você deve seguir essas etapas:
- Carregar os dados tabulares no Google Colab.
- Tratar a formatação dos atributos numéricos e categóricos.
- Lidar com valores faltantes e redundantes, outliers e ruído.
- Normalizar ou padronizar os dados.

<b>Solução Atividade 1:</b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Dados_Hepatite_Ativ_01.ipynb" target="_blanck"><i><u>Dados_Hepatite_Ativ_01.ipynb</u></i></a>
</br></br>
<b> Atividade 2: Preparação final do conjunto de dados de hepatite para processamento com modelo preditivo </b>
<p>Você, como cientista de dados do Hospital Santa Ajuda, completará a preparação do conjunto de dados de hepatite, com uma etapa de seleção de características. O objetivo desta atividade é a utilização de métodos de seleção de características e redução de dimensionalidade para elaborar três conjuntos de dados com quantidade de atributos distinta do original. Os cenários são definidos a seguir:</p>

- 1° Cenário: Seleção dos 12 melhores atributos utilizando correlação como base;
- 2° Cenário: Seleção dos 12 melhores atributos utilizando chi quadrado como base;
- 3° Cenário: Redução de dimensionalidade com PCA do conjunto total para 12 componentes principais.

<p>Uma vez que os dados estejam prontos para cada cenário, a experimentação com modelos de classificação pode ser feita utilizando um modelo de máquinas de vetor suporte (SVM) com o kernel RBF e avaliando a métrica de acurácia balanceada, uma vez que o conjunto original apresenta mais dados de pessoas que sobreviveram do que para aquelas que vieram a óbito.</p>

<b>Solução Atividade 2:</b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Dados_Hepatite_Ativ_02.ipynb" target="_blanck"><i><u>Dados_Hepatite_Ativ_02.ipynb</u></i></a>

<hr>

#### Situação-problema2: Contextualização

<b>Soluções para o ramo hoteleiro</b>

  Uma empresa do ramo hoteleiro está passando por algumas dificuldades em seu sistema. Por isso, ela contratou a empresa de IA que você trabalha para elaborar soluções específicas utilizando ciência de dados para as necessidades dela. Logo de início, ela especificou que gostaria da elaboração de soluções “prova de conceito” para dois desafios distintos:
- Sistema de reconhecimento de face para biometria de funcionários; 
- Previsão da nota geral de avaliação de um hóspede com base no que ele escreveu sobre o hotel na plataforma TripAdvisor.
  Ambos são problemas que necessitam do trabalho com dados não estruturados, logo, uma etapa de extração de características será necessária.

<b> Atividade 1: Sistema de reconhecimento </b>

  Lembre-se de que você, funcionário de uma empresa de IA, deve elaborar um sistema de reconhecimento de face para biometria de funcionários, de modo a atender à empresa do ramo hoteleiro.
  Para tal, a prova de conceito deverá ser desenvolvida utilizando o Yale Face Dataset, um dataset com 165 imagens em preto e branco do rosto de 15 pessoas diferentes (11 imagens por pessoa). O arquivo .zip da base pode ser baixado através do seguinte link: <a href="" target="_blanck"><i><u>http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip</u></i></a>.
  O objetivo central da atividade é extrair um vetor das características que melhor representem as nuances de cada rosto. Como são 165 imagens, o dataset final esperado pós-processamento será uma matriz de 165 x N, onde N é a quantidade de características escolhidas para representação.

Solução da Atividade 1: <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Sistema_de_reconhecimento.ipynb" target="_blanck"><i><u>Sistema de reconhecimento.ipynb</u></i></a>

<b> Atividade 2: Previsão de nota </b>

  Você, funcionário de uma empresa de IA, deve realizar uma previsão da nota geral de avaliação de um hóspede com base no que ele escreveu sobre o hotel na plataforma TipAdvisor, de modo a atender à empresa do ramo hoteleiro.
  Para efetuá-la, a viabilidade de uma solução deverá utilizar o conjunto de dados disponível online na plataforma Kaggle, onde 20 mil avaliações foram extraídas com as suas respectivas notas gerais. O arquivo .zip da base pode ser baixado através do link: <a href="" target="_blanck"><i><u>https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews/download</u></i></a>.
  O objetivo central da atividade é organizar o vocabulário utilizado nas avaliações e representar o texto utilizando a estratégia de bag-of-words. Ao fim, como são 20491 amostras, é esperado que no pós-processamento você tenha uma matriz com 20491 x N, onde N é a quantidade de características escolhidas para representação do documento.

<b>Solução da Atividade 2:</b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Previs%C3%A3o_de_nota.ipynb" target="_blanck"><i><u>Previsão de nota.ipynb</u></i></a>

<hr>

### Criação e validação de modelos de aprendizado de máquina

#### Situação-problema1: Contextualização

<b> Análise sobre imóveis </b>
 
Você está terminando sua formação no curso de Criação e validação de modelos de aprendizado de máquina. Ao buscar por empregos na área, você conseguiu uma entrevista para demonstrar seus conhecimentos na análise e criação de modelos de machine learning.
 
Nesta, o entrevistador lhe pediu para correlacionar características de imóveis, como número de quartos e renda dos proprietários, com seus valores de mercado. A base de dados a ser utilizada contém o levantamento de 20.640 imóveis, a partir do Censo de 1990. O conjunto de dados foi gerado pelos pesquisadores R. Kelley Pace e Ronald Barry, em 1997 (KELLEY PACE & BARRY, 1997).

<b>Atividade 1: Análise e criação de modelo supervisionado</b> 

Seu entrevistador lhe pediu para correlacionar características de imóveis, como número de quartos e renda dos proprietários, com seus valores de mercado. A base de dados a ser utilizada contém o levantamento de 20.640 imóveis, a partir do Censo de 1990. O conjunto de dados foi gerado pelos pesquisadores R. Kelley Pace e Ronald Barry, em 1997 (KELLEY PACE & BARRY, 1997).

Você poderá acessar os dados gratuitamente no site dos autores. Apesar do formato não ser intuitivo em um primeiro momento, os arquivos podem ser abertos por editores de texto convencionais e salvos como “.CSV”. O link para acesso à base está indicado abaixo:

Disponível em: <a href="https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html" target="_blanck"><i><u>https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html</u></i></a>

Vale dizer que a biblioteca Scikit-Learn contém nativamente esta base de dados, permitindo que seja importada automaticamente para a aplicação. Mais informações e exemplos podem ser obtidos no seguinte link:

   Disponível em: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html" target="_blanck"><i><u>https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html</u></i></a>

Assim, recomenda-se o uso biblioteca Scikit-Learn para solução do desafio. Bibliotecas adicionais, como Pandas, Numpy e Matplotlib, também podem ser utilizadas.

<b>Solução Atividade 1:</b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Previs%C3%A3o_pre%C3%A7os_im%C3%B3veis.ipynb" target="_blanck"><i><u>Previsão preços imóveis.ipynb</u></i></a>
</br></br>

#### Situação-problema 2: Contextualização

Você assumiu o posto de cientista de dados de um Centro de Pesquisa renomado voltado à área botânica. No caso, os pesquisadores do centro estão realizando um estudo sobre espécies de flores, analisando suas pétalas e sépalas.
 
Como foi gerado muitos dados, foi-lhe pedido para analisá-los, de modo a classificar as espécies. Assim, seu desafio consiste em:

• Criar uma solução de agrupamento utilizando os algoritmos adequados.

<b> Atividade 1: Análise e criação de modelo para classificar flores </b>

Nesta atividade, você auxiliará um grupo de pesquisadores quanto à classificação de espécies de flores. Você se baseará nas informações das pétalas e das sépalas delas.

A base de dados a ser utilizada foi criada em 1936 por R. A. Fisher, para separação de espécie de flor Íris (FISHER, 1936). Ela contém 150 amostras, divididas em três espécies diferentes, cada uma com 50 amostras. Vale dizer que algumas dessas amostras são facilmente separáveis e classificáveis, enquanto outras não podem ser classificadas de forma fácil.

Para acesso à base de dados, é possível utilizar dois métodos diferentes. Você poderá acessar o arquivo “.csv”, gratuitamente, por meio da plataforma Kaggle, no seguinte link:

Disponível em: <a href="https://www.kaggle.com/uciml/iris" target="_blanck"><i><u>https://www.kaggle.com/uciml/iris</u></i></a>

A biblioteca Scikit-Learn contém nativamente esta base de dados, permitindo que seja importada automaticamente para a aplicação. Mais informações e exemplos podem ser obtidos no seguinte link:

Disponível em: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html" target="_blanck"><i><u>https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html</u></i></a>

<b> Solução Atividade 1: </b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Classifica%C3%A7%C3%A3o_de_esp%C3%A9cies_de_flores_(%C3%8Dris).ipynb" target="_blanck"><i><u>Classificação de espécies de flores (Íris).ipynb</u></i></a>

<hr>

### Plataformas de Machine Learning

<b> Atividade 1: Classificação e Regressão com Scikit-learn</b>

Etapa 1

Usando o dataset a seguir, programe um modelo de regressão linear para prever a nota de um aluno com base no número de horas de estudo. Avalie a performance do modelo mostrando a sua acurácia.

Etapa 2

Crie modelos, que irão classificar se uma transação é fraude ou não, com os seguintes classificadores:

- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosted Trees Regressor
- Ensemble Classifier

Compare a acurácia de cada classificador e mostre em qual obtivemos o melhor resultado.

<b> Solução Atividade 1: </b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Classifica%C3%A7%C3%A3o_e_Regress%C3%A3o_com_Scikit_learn.ipynb" target="_blanck"><i><u>Classificação e Regressão com Scikit-learn.ipynb</u></i></a>

<b> Atividade 2: TensorFlow</b>

Utilizando o dataset Iris, faça um simples classificador binário para prever se uma flor é a espécie Iris setosa ou não. Em outras palavras, este conjunto de dados tem três espécies, mas apenas preveja se uma flor é uma única espécie, Iris setosa ou não.

O Iris Dataset é disponível no link: <a href="https://www.tensorflow.org/datasets/catalog/iris" target="_blanck"><i><u>https://www.tensorflow.org/datasets/catalog/iris</u></i></a>

<b> Solução Atividade 2: </b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/TensorFlow_Iris.ipynb" target="_blanck"><i><u>TensorFlow:Iris.ipynb</u></i></a>

<b> Atividade 3: Keras e PyTorch</b>

Etapa 1- Keras

Uma de suas tarefas é voltada para área da saúde, foi-lhe pedido para prever se um câncer é benigno ou maligno.

No caso, desenvolva um algoritmo que use redes neurais para prever, com precisão (~94% de precisão), se um tumor de câncer de mama é benigno ou maligno, basicamente ensine uma máquina a prever o câncer de mama. Para tal, uma sugestão é utilizar APIs Keras.

A seguir, será detalhado o conjunto de dados:

Dataset do câncer de mama: as características são computadas a partir de uma imagem digitalizada de uma agulha fina aspirada (FNA) de uma massa mamária. Eles descrevem características dos núcleos celulares presentes na imagem. Algumas das imagens estão disponíveis em: <a href="http://www.cs.wisc.edu/~street/images" target="_blanck"><i><u>http://www.cs.wisc.edu/~street/images</u></i></a> .

- Distribuição da classe: 357 benigno, 212 maligno.

Dataset disponível no link: <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29" target="_blanck"><i><u>https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29</u></i></a>

Etapa 2 - PyTorch

Utilizando PyTorch, construa os modelos dos casos abaixo utilizando a mesma base da atividade anterior

- Classificação binária utilizando o Iris Dataset
- Previsão de fraudes V2
- Classificador do câncer de mama
- Marketing Bancário

<b> Solução Atividade 3: </b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Keras_e_PyTorch.ipynb" target="_blanck"><i><u>Keras_e_PyTorch.ipynb</u></i></a>

<b> Atividade 4: Convolutional Neural Networks (CNNS)</b>

Implemente uma rede LeNet5 para detectar dígitos manuscritos. Utilize o MNIST dataset para treinar o modelo.

<b> Solução Atividade 4: </b> <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Convolutional_Neural_Networks_(CNNS).ipynb" target="_blanck"><i><u>Convolutional Neural Networks (CNNS).ipynb</u></i></a>

<b> Atividade 5: Transfer Learning </b>

Construa um classificador de imagens usando a rede AlexNet pré-treinada que distingue entre cães e gatos. 
Utilize o dataset, disponível em: <a href="https://www.kaggle.com/c/dogs-vs-cats/data" target="_blanck"><i><u>https://www.kaggle.com/c/dogs-vs-cats/data</u></i></a>.
Esse dataset contém 20.000 imagens rotuladas e os conjuntos de teste e validação têm 2.500 imagens. Para utilizá-lo, você deve remodelar cada imagem para 227×227×3.

<b> Solução Atividade 5: </b> <a href="" target="_blanck"><i><u>Transfer Learning.ipynb</u></i></a>

<hr>
