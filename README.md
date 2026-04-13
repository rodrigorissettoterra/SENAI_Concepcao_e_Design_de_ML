# SENAI: Concepção e Design de Machine Learning
## Exercícios realizados no curso do Senai: Concepção e Design de Machine Learning

### Preparação e pré-processamento de dados

### Situação-problema1: Contextualização

<b>Dados sobre hepatite</b>

  Você assumiu o posto de cientista de dados do Hospital Santa Ajuda e lhe foi pedido que auxilie os médicos na identificação de fatores prognósticos e sobrevivência dos pacientes com hepatite. No caso, trabalhará no pré-processamento de um conjunto de dados que envolvem esses pacientes para criação de um modelo preditivo. 
  A base de dados a ser utilizada foi disponibilizada primeiramente em 1988 (GONG, 1988) no repositório da UCI e é bem estabelecida na literatura como um projeto de entrada no mundo da ciência de dados (GOH, 2020). A base apresenta 19 atributos de 155 pacientes, que incluem dados sobre o seu perfil, presença de sinais e sintomas da doença, resultados
de exames e testes em laboratório, e protocolo de tratamento adotado, além de incluir o rótulo indicando se o paciente com a doença veio a falecer. 
  Após o pré-processamento deste conjunto de dados, é esperado que eles não possuam inconsistências, redundâncias, valores faltantes, ruído e que estejam normalizados para serem utilizados como entrada por um modelo de aprendizado de máquina.
  Desse modo, você deverá superar o seguinte desafio:
• Preparar o conjunto de dados sobre hepatite.

</br>

<b> Atividade 1: Preparação inicial do conjunto de dados de hepatite

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

Solução Atividade 1: <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Dados_Hepatite_Ativ_01.ipynb" target="_blanck"><i><u>Dados_Hepatite_Ativ_01.ipynb</u></i></a>
</br></br>
<b> Atividade 2: Preparação final do conjunto de dados de hepatite para processamento com modelo preditivo
  Você, como cientista de dados do Hospital Santa Ajuda, completará a preparação do conjunto de dados de hepatite, com uma etapa de seleção de características.
  O objetivo desta atividade é a utilização de métodos de seleção de características e redução de dimensionalidade para elaborar três conjuntos de dados com quantidade de atributos distinta do original. Os cenários são definidos a seguir:
- 1° Cenário: Seleção dos 12 melhores atributos utilizando correlação como base;
- 2° Cenário: Seleção dos 12 melhores atributos utilizando chi quadrado como base;
- 3° Cenário: Redução de dimensionalidade com PCA do conjunto total para 12 componentes principais.
  Uma vez que os dados estejam prontos para cada cenário, a experimentação com modelos de classificação pode ser feita utilizando um modelo de máquinas de vetor suporte (SVM) com o kernel RBF e avaliando a métrica de acurácia balanceada, uma vez que o conjunto original apresenta mais dados de pessoas que sobreviveram do que para aquelas que vieram a óbito. 

Solução Atividade 2: <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Dados_Hepatite_Ativ_02.ipynb" target="_blanck"><i><u>Dados_Hepatite_Ativ_02.ipynb</u></i></a>

<hr>

### Situação-problema2: Contextualização

<b>Soluções para o ramo hoteleiro</b>

  Uma empresa do ramo hoteleiro está passando por algumas dificuldades em seu sistema. Por isso, ela contratou a empresa de IA que você trabalha para elaborar soluções específicas utilizando ciência de dados para as necessidades dela. Logo de início, ela especificou que gostaria da elaboração de soluções “prova de conceito” para dois desafios distintos:
- Sistema de reconhecimento de face para biometria de funcionários; 
- Previsão da nota geral de avaliação de um hóspede com base no que ele escreveu sobre o hotel na plataforma TripAdvisor.
  Ambos são problemas que necessitam do trabalho com dados não estruturados, logo, uma etapa de extração de características será necessária.

<b> Atividade 1: Sistema de reconhecimento

  Lembre-se de que você, funcionário de uma empresa de IA, deve elaborar um sistema de reconhecimento de face para biometria de funcionários, de modo a atender à empresa do ramo hoteleiro.
  Para tal, a prova de conceito deverá ser desenvolvida utilizando o Yale Face Dataset, um dataset com 165 imagens em preto e branco do rosto de 15 pessoas diferentes (11 imagens por pessoa). O arquivo .zip da base pode ser baixado através do seguinte link: <a href="" target="_blanck"><i><u>http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip</u></i></a>.
  O objetivo central da atividade é extrair um vetor das características que melhor representem as nuances de cada rosto. Como são 165 imagens, o dataset final esperado pós-processamento será uma matriz de 165 x N, onde N é a quantidade de características escolhidas para representação.

Solução da Atividade 1: <a href="https://github.com/rodrigorissettoterra/SENAI_Concepcao_e_Design_de_ML/blob/main/Sistema_de_reconhecimento.ipynb" target="_blanck"><i><u>Sistema de reconhecimento.ipynb</u></i></a>

<b> Atividade 2: Previsão de nota

  Você, funcionário de uma empresa de IA, deve realizar uma previsão da nota geral de avaliação de um hóspede com base no que ele escreveu sobre o hotel na plataforma TipAdvisor, de modo a atender à empresa do ramo hoteleiro.
  Para efetuá-la, a viabilidade de uma solução deverá utilizar o conjunto de dados disponível online na plataforma Kaggle, onde 20 mil avaliações foram extraídas com as suas respectivas notas gerais. O arquivo .zip da base pode ser baixado através do link: <a href="" target="_blanck"><i><u>https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews/download</u></i></a>.
  O objetivo central da atividade é organizar o vocabulário utilizado nas avaliações e representar o texto utilizando a estratégia de bag-of-words. Ao fim, como são 20491 amostras, é esperado que no pós-processamento você tenha uma matriz com 20491 x N, onde N é a quantidade de características escolhidas para representação do documento.

Solução da Atividade 2: <a href="" target="_blanck"><i><u>NOME</u></i></a>
