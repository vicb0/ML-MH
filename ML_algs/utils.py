import os
def save_results(file_name:str, title:str, content:list):
    os.makedirs('./results', exist_ok=True)

    f = open(f"./results/{file_name}.txt", "a")
    f.write(f'{title}\n')
    for c in content:
        f.write(f'{c}\n')
    print(f"Escrita dos resultados de {title} realizada\n")
    f.close()

