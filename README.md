As pastas contidas no .gitignore são muito grandes para o GitHub.

O MH-100k deve ser baixado pelo link https://figshare.com/articles/dataset/MH-100K-Dataset/24328885 e extraído.

As outras pastas serão geradas ao rodar o script parser.py, o que leva algum tempo para finalizar.

O parser dividide os ~90k apks benignos em fragmentos menores, com cada um tendo aproximadamente o mesmo tamanho do fragmento com os ~10k malignos.

Além disso, o parser gera os fragmentos no formato .h5, que é mais eficiente de armazenar e carregar para os algoritmos, compostos por 10k benignos e 10k malignos, ou seja, já balanceados para o treinamento dos modelos.

Após rodar o `parser.py`, rode o `compression.py`.
Este script irá gerar um csv com as variancias de todas as colunas (utilizado para treinar o modelo), e um .h5 do dataset MH-100K inteiro.

Enfim, vá para o arquivo `main.py` e siga as instruções lá presentes para baixar o dataset MH-1M, utilizado para verificar os resultados do modelo treinado utilizando o dataset MH-100K.
