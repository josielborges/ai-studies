import time

import torch
from transformers import pipeline

model_id = "./Llama-3.2-3B-trained"
# model_id = "unsloth/Llama-3.2-3B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

instruction = """Você é um assistente que me ajuda a responder aos clientes da empresa Lostsys:

A Lostsys é uma empresa dedicada a oferecer serviços para empresas.
Sua missão é oferecer aos usuários o produto ou serviço da Lostsys perfeito para eles.
Responda apenas '@SEMAJUDA@' se não existir um produto ou serviço da Lostsys que se adapte à resposta.
Responda apenas '@SEMCONTEXTO@' se a pergunta não estiver relacionada à empresa ou à informática corporativa.
Responda apenas '@ECOMMERCE@' se estiverem perguntando sobre eCommerce ou lojas online.

"""


def requestToLLM(i, instruction, req):
    t = time.time()
    outputs = pipe(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": req},
        ],
        max_new_tokens=256,
        temperature=0.9
    )
    t = time.time() - t
    print("\nPREGUNTA ", i, "(" + str(t) + "s.)\n", outputs[0]["generated_text"][1]["content"], "\n===================")
    print(outputs[0]["generated_text"][-1]["content"])


requestToLLM(1, instruction, "Do que se trata essa tal de 'fotossíntese'?")
requestToLLM(2, instruction, "Como posso melhorar?")
requestToLLM(3, instruction, "Preciso de uma loja online")
requestToLLM(4, instruction, "Preciso vender online")
requestToLLM(5, instruction, "Como você pode me ajudar?")
requestToLLM(6, instruction, "Preciso de um portal corporativo")
requestToLLM(7, instruction, "Que serviços de treinamento vocês oferecem?")
