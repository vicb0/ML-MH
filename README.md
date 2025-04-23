As pastas contidas no .gitignore são muito grandes para o GitHub.

O MH-100K deve ser baixado pelo link https://figshare.com/articles/dataset/MH-100K-Dataset/24328885 e as partes devem ser extraídas utilizando Winrar.

O Mh-100M deve ser baixado seguindo o seguinte passo-a-passo:
- Rode o comando: `git clone https://github.com/Malware-Hunter/MH-1M`
- Vá até `data/compressed/zip-intents-permissions-opcodes-apicalls`
- Rode o comando `copy /b amex-1M-[intents-permissions-opcodes-apicalls].npz.7z.part* full_archive.7z`
- Extraia full_archive.7z usando 7zip, renomeie o .npz para dataset.npz

As outras pastas serão geradas ao rodar os scripts `parser[1m, 100k].py`, o que leva algum tempo para finalizar.

O parser dividide os apks benignos em fragmentos menores.

Além disso, o parser gera os fragmentos no formato .h5, que é mais eficiente de armazenar e carregar para os algoritmos já balanceados para o treinamento dos modelos.

O `compression.py`
Este script irá gerar um .h5 do dataset MH-100K inteiro.

# Rode o arquivo `main.py`
O script `main` irá rodar todos os passos acima automaticamente, com exceção do download dos dois datasets.
