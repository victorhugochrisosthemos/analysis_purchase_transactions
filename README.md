# Link para o site - >

### Como o .csv precisa estar?

- O arquivo precisa ter essas 3 colunas exatamente com esses nomes: CustomerID, Products, Timestamp
<br>
### CustomerID 
<br>
O que é: identificador único do cliente
<br>
Formato aceito:
<br>
número (101, 202, 303) ou texto (C_001, ABC123)
<br>
Regras:
<br>
não pode ficar vazio (sem NaN)
<br>
não precisa estar em ordem
<br>
pode se repetir (porque um cliente aparece em várias compras)
<br>
Exemplo:
<br>
CustomerID<br>
101<br>
101<br>
205<br>
205<br>
205<br>

### Products 
<br>
O que é: uma string com produtos separados por vírgula, representando os itens daquela compra/transação.
<br>
Formato obrigatório:
<br>
produtos separados por vírgula
<br>
idealmente com espaço opcional após a vírgula
<br>
ex.: "Milk, Cereal, Cookie"
<br>
Regras importantes:
<br>
o separador precisa ser vírgula ,
<br>
não usar ponto e vírgula ; 
<br>
evitar produtos com vírgula no nome (ex.: "Molho, Tomate"). Se existir, tem que padronizar o nome sem vírgula.
<br>
Exemplos válidos:
<br>
Milk, Cereal, Cookie
<br>
Bread, Banana
<br>
Egg
<br>
Exemplos que vai dar ruim:
<br>
Milk; Cereal; Cookie (separador errado)
<br>
Milk | Cereal | Cookie (separador errado)
<br>
vazio (transação sem produto)
<br>
Dica de padronização (muito recomendada):
<br>
manter a mesma grafia sempre (ex.: não misturar Milk e milk)
<br>
remover espaços duplos
<br>
evitar acentos diferentes para o mesmo produto
<br>


### Timestamp 
<br>
O que é: data e/ou data+hora de quando a compra ocorreu.
<br>
Formato aceito pelo código (pandas to_datetime):
<br>
YYYY-MM-DD → 2025-01-30
<br>
YYYY-MM-DD HH:MM:SS → 2025-01-30 14:35:10
<br>
DD/MM/YYYY → 30/01/2025 (normalmente funciona, mas às vezes depende do padrão)
<br>
DD/MM/YYYY HH:MM → 30/01/2025 14:35
<br>
Regras:
<br>
não pode ter valores inválidos (ex.: 32/13/2025)
<br>
linhas com data inválida são descartadas no módulo temporal (porque vira NaT e o código dá dropna)
<br>
Exemplos válidos:
<br>
2024-11-05 10:13:00
<br>
2024-11-05
<br>
05/11/2024 10:13
<br>
Cada linha do CSV representa uma transação/compra
<br>
Exemplo ideal:
<br>
CustomerID	Products	Timestamp<br>
101	Milk, Cereal, Cookie	2024-01-05 09:10:00<br>
101	Bread, Egg	2024-01-10 18:20:00<br>
205	Milk, Banana	2024-02-01 14:00:00<br>
<br><br>

#### Exemplo de template pronto
CustomerID,Products,Timestamp
101,"Milk, Cereal, Cookie",2024-01-05 09:10:00
101,"Bread, Egg",2024-01-10 18:20:00
205,"Milk, Banana",2024-02-01 14:00:00

