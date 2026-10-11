# Laboratório pessoal de Motion Transfer e Object Swap

[Abrir o notebook diretamente no Google Colab](https://colab.research.google.com/github/reedzinn/genjutsu/blob/main/notebooks/Genjutsu_Pessoal_Colab.ipynb)

Protótipo próprio com **Wan2.1-VACE-1.3B**, sem API paga, sem treinamento e sem ligação com o produto Higgsfield Genjutsu.

**Estado:** interface e preparação podem ser testadas sem GPU. A inferência real ainda precisa ser validada em GPU NVIDIA. Não há promessa de qualidade igual ao Genjutsu nem de compatibilidade com a RX 580.

## Seu computador: Windows, RX 580 2048SP 8 GB, 16 GB RAM

O navegador exibe a interface; ele não substitui a GPU que executa o modelo. A RX 580 não faz parte das arquiteturas atuais indicadas no caminho oficial ROCm/ComfyUI para Windows. O código atual do ComfyUI também alerta que torch-directml funciona mal e é lento. Por isso este protótipo não tenta instalar CUDA na Radeon nem apresenta DirectML como solução validada.

No seu PC, a interface local permite enviar arquivos e conferir as regiões. **A geração foi deliberadamente condicionada a CUDA/NVIDIA**: não é uma afirmação de que toda execução em AMD seja impossível, e sim o limite do caminho implementado.

## Caminho gratuito pelo navegador

1. Abra o Google Colab e use **Arquivo → Fazer upload de notebook** para abrir `notebooks/Genjutsu_Pessoal_Colab.ipynb`.
2. Se sua conta receber uma GPU gratuita, selecione-a em **Ambiente de execução → Alterar tipo de ambiente de execução**. Se não houver GPU gratuita, pare; o programa não compra créditos ou contrata máquina alguma.
3. Execute as células em ordem. O notebook já inclui o código: não depende de publicar o repositório.
4. Envie um vídeo e uma imagem. Comece com 17 quadros, aproximadamente um segundo, em 256×144.
5. Escolha `motion` ou `swap`, confira a prévia e execute a geração.
6. Baixe o vídeo e `run.json` antes de encerrar a sessão. O armazenamento da sessão é temporário.

A disponibilidade de GPU gratuita não é garantida. RAM, VRAM e tempo precisam ser medidos na sessão real. Os modelos ocupam vários GB; o notebook verifica espaço antes da instalação. Os arquivos enviados são processados na nuvem da sessão, não no PC. Não abre túnel nem link público Gradio.

## O que foi implementado

- **Motion Transfer:** detector OpenPose extrai pose corporal; VACE recebe sequência de poses e imagem de referência. Sem transferência facial ou rastreamento de múltiplas pessoas.
- **Object Swap:** VACE recebe vídeo, imagem de referência e máscaras retangulares. Uma região é fixa; duas regiões interpolam uma trajetória linear. Não há segmentação ou rastreamento inteligente neste primeiro protótipo.
- Prévia de região, limites de upload, redução de resolução, amostragem temporal e ajuste ao número de quadros aceito pelo VACE.
- Codificador de texto em 4 bits, liberado antes de carregar o gerador; transferência de partes do gerador entre RAM/GPU para reduzir memória. A configuração é experimental e pode alterar qualidade.
- Progresso real por etapas, MP4 sem áudio e registro de modelo, versões, parâmetros, hashes, tempo, estado e erros.
- Ausência de GPU produz erro explícito. Não há vídeo de demonstração apresentado como resultado gerado.

Objetos parcialmente escondidos, detalhes pequenos e grandes mudanças de forma podem falhar. O fundo pode mudar mesmo fora da máscara: não é uma composição com preservação perfeita dos pixels originais. Os rascunhos em baixa resolução servem para validar o fluxo, não demonstram qualidade final.

## Interface local no Windows

Instale Python 3.11 ou 3.12 e abra um terminal na pasta do projeto:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

Abra no navegador o endereço local informado pelo programa. Não é necessário executar como administrador. Essa instalação prepara **somente a interface e as prévias** na sua máquina. Não habilita geração na RX 580.

Para uma máquina com CUDA/NVIDIA já configurada, use um ambiente virtual, instale `requirements-model.txt` e execute `app.py`. Não instale o motor pesado no seu Windows/Radeon esperando que CUDA passe a funcionar.

## Desenvolvimento e verificação

```bash
python -m pip install -r requirements.txt pytest==8.4.2 nbformat==5.10.4
python -m pytest -q
python scripts/build_notebook.py
```

Os testes cobrem vídeos reais de teste, amostragem, máscaras, exportação e recusa de gerar sem GPU. Não substituem um teste de inferência com os pesos. Alterações em `engine.py`, `media.py`, `app.py` ou dependências exigem reconstruir o notebook antes de distribuí-lo.

## Fontes

- [VACE: funções e limitações](https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/UserGuide.md).
- [VACE: modelos, dimensões e licenças](https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/README.md).
- [Pipeline Diffusers utilizado](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/wan/pipeline_wan_vace.py).
- [ComfyUI: suporte AMD/Windows](https://github.com/Comfy-Org/ComfyUI#amd-gpus-windows-rocm-100).
- [ComfyUI: aviso sobre DirectML](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py).
- [Microsoft: alcance geral do DirectML](https://github.com/microsoft/DirectML/blob/master/PyTorch/README.md). Suporte geral a DirectX12 não comprova funcionamento deste modelo.
- [Colab: informações e FAQ oficiais](https://github.com/googlecolab/colabtools).

Modelos e componentes mantêm suas próprias licenças. Este protótipo foi preparado para o uso pessoal solicitado. Software gratuito não fornece automaticamente computação gratuita.
