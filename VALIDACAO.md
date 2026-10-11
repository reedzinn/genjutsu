# Verificação desta versão

Ambiente de teste: Linux, Python 3.12, PyTorch 2.7.1 CPU, sem GPU disponível.

## Executado

- 11 testes passaram: leitura e amostragem de MP4, interpolação e validação de máscaras, exportação de vídeo, arquivo inválido e recusa explícita de geração sem GPU.
- Interface Gradio iniciada; página e configuração retornaram HTTP 200.
- Upload real de vídeo de teste pela API da interface; duas prévias de região retornadas em 256×144.
- Pedido de geração pela interface recusado com erro claro pela ausência de GPU; nenhum vídeo de resultado falso criado.
- Dependências do motor instaladas sem baixar pesos; imports de VACE, UMT5, quantização e detector OpenPose verificados.
- Assinaturas da versão fixada do VACE conferidas para vídeo, máscara, imagem de referência, embeddings de texto e callback.
- `pip check` sem conflitos no ambiente de teste final.
- Notebook validado como formato Jupyter; todas as células de código compiladas.

## Ainda não executado

- Instalação no Windows do usuário.
- Geração de um vídeo usando os pesos reais em CUDA, incluindo codificação de texto em 4 bits.
- Execução do notebook no Colab ou confirmação de GPU gratuita disponível na conta do usuário.
- Medição de VRAM, RAM, duração de inferência ou qualidade de Motion Transfer/Object Swap.

Esses resultados validam a preparação e a integração estrutural, não a geração de vídeo. A primeira transformação real é o próximo critério de aceitação. O programa não oferece inferência na RX 580 nesta versão.
