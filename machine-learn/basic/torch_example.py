# Este programa utiliza o framework PyTorch para criar uma rede neural
# capaz de aprender com dados e fazer previsões.

# Primeiro, importamos as bibliotecas necessárias
import torch
import torch.nn as nn
import torch.optim as optim

# Em seguida, criamos os dados de entrada e saída
x = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

# Em seguida, criamos a classe da rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # A rede neural tem 2 camadas ocultas com 5 neurônios cada
        # A entrada tem 1 neurônio e a saída tem 1 neurônio
        self.fc1 = nn.Linear(1, 5)  # De 2 para 1 na entrada
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        # A função forward define como a rede neural processa as entradas
        x = torch.relu(self.fc1(x))  # Ativação ReLU na primeira camada oculta
        x = self.fc2(x)  # Segunda camada oculta
        return x

# Instanciamos a rede neural
model = Net()

# Definimos a função de perda (erro) que a rede neural deve minimizar
criterion = nn.MSELoss()

# Exibe os parâmetros da rede neural
print(model.parameters())

# Definimos o otimizador que irá atualizar os parâmetros da rede neural
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamos a rede neural por 1000 épocas
for epoch in range(1000):
    # Zeramos os gradientes
    optimizer.zero_grad()
    # Calculamos a saída da rede neural
    outputs = model(x)
    # Calculamos o erro da rede neural
    loss = criterion(outputs, y)
    # Calculamos os gradientes
    loss.backward()
    # Atualizamos os parâmetros da rede neural
    optimizer.step()

    # Exibimos o erro a cada 100 épocas
    if epoch % 100 == 99:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Fazemos uma previsão com a rede neural treinada
with torch.no_grad():
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')