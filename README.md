## Sistema de visão computacional e machine learning para verificação de disponibilidade de vagas de estacionamento.
### Automated system to parking lot using computer vision (opencv, machine learning)

## Execução (Run)

Primeiro instale as bibliotecas necessárias, recomendo utilizar o python3.6 que foi a versão usada no desenvolvimento:

> pip install -r req.txt

Pronto, agora basta executar o arquivo app.py:

> python app.py

## Overview

Basicamente primeiro é extraído as features do local a ser analisado como histograma, bordas o que alimenta um algoritmo SVM para realizar a classificação. Realizei o procedimento somente em duas vagas, mas poderia ser aplicado a todas a vagas.





