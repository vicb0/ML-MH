As pastas contidas no .gitignore são muito grandes para o GitHub.
O MH-100k deve ser baixado pelo link https://figshare.com/articles/dataset/MH-100K-Dataset/24328885 e extraído.
As outras pastas serão geradas ao rodar o script parser.py, o que leva algum tempo para finalizar.

1. O parser dividide os ~90k apks benignos em fragmentos menores, com cada um tendo aproximadamente o mesmo tamanho do fragmento com os ~10k malignos.
2. Em seguida, remove as colunas em que todos os registros daquele fragmento em específico tenham todos o mesmo valor.
