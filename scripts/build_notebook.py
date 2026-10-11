"""Build a self-contained notebook: no public GitHub push or secret required."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text):
    text = textwrap.dedent(text).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
              "id": hashlib.sha256(text.encode()).hexdigest()[:12]}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


files = {name: (ROOT / name).read_text() for name in (
    "media.py", "engine.py", "app.py", "requirements.txt", "requirements-model.txt")}
cells = [
    cell("markdown", """
    # Laboratório pessoal: Motion Transfer e Object Swap

    Protótipo com **Wan2.1-VACE-1.3B**, sem API paga. A inferência ainda não foi
    validada em GPU neste projeto; podem ser necessários ajustes de memória ou compatibilidade.

    No Colab gratuito: **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**, se disponível.
    A disponibilidade e o tempo gratuito dependem do Google. Se não houver GPU gratuita,
    pare: este notebook não compra créditos nem cria máquinas pagas. Ele não elimina limites do Colab.

    Seu Windows/RX 580 serve para acessar o navegador. A GPU que gera o vídeo é a da sessão Colab.
    Os arquivos enviados ficam na sessão de nuvem; não são processados na RX 580.
    Reserve espaço para vários GB de modelos e comece com 17 quadros (~1 segundo).

    Execute as células na ordem. Não é necessário publicar o repositório ou fornecer chaves de API.
    """),
    cell("code", """
    import os, shutil
    import torch
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise RuntimeError("Esta sessão não tem GPU NVIDIA. Se o Colab não oferecer uma gratuitamente, pare aqui.")
    if tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2, 5):
        raise RuntimeError("Esta sessão usa PyTorch antigo. Use uma sessão Colab atual; não altere os drivers manualmente.")
    if shutil.disk_usage('/content').free < 25 * 1024**3:
        raise RuntimeError("Libere pelo menos 25 GB no disco da sessão antes de baixar os modelos.")
    print('GPU da sessão:', torch.cuda.get_device_name())
    print('Memória da GPU (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    print('A geração será testada nesta GPU, não na placa do seu computador.')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HOME'] = '/content/genjutsu-model-cache'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    """),
    cell("markdown", "## Preparar o programa e instalar as dependências gratuitas"),
    cell("code", "from pathlib import Path\nimport json, os\n"
         "root = Path('/content/genjutsu-pessoal')\nroot.mkdir(exist_ok=True)\n"
         + "embedded = json.loads(" + repr(json.dumps(files, ensure_ascii=False)) + ")\n"
         + "for name, source in embedded.items():\n"
           "    destination = root / name\n"
           "    if destination.exists() and destination.read_text() != source:\n"
           "        raise RuntimeError(f'{destination} contém alterações. Preserve-as antes de substituir.')\n"
           "    destination.write_text(source)\n"
           "os.chdir(root)\nprint('Programa preparado em', root)\n"),
    cell("code", """
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-model.txt'], check=True)
    from importlib.metadata import version
    changed = []
    for module_name, package_name in [('numpy', 'numpy'), ('PIL', 'Pillow'), ('torch', 'torch'),
                                      ('huggingface_hub', 'huggingface-hub')]:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, '__version__', None) != version(package_name):
            changed.append(package_name)
    if changed:
        raise RuntimeError('Dependências atualizadas: ' + ', '.join(changed) +
                           '. Reinicie a sessão (sem excluir os arquivos) e execute as células novamente.')
    # O notebook não inicia servidores nem túneis públicos.
    print('Dependências instaladas. A geração ainda precisa ser executada.')
    """),
    cell("markdown", "## Enviar um vídeo e uma imagem\nEnvie arquivos próprios. O resultado inicial não terá áudio."),
    cell("code", """
    from google.colab import files
    from pathlib import Path
    import uuid
    upload_dir = root / 'uploads' / uuid.uuid4().hex
    upload_dir.mkdir(parents=True)

    def upload_one(label, allowed):
        print(label)
        received = files.upload()
        if len(received) != 1:
            raise ValueError('Envie exatamente um arquivo por vez.')
        name, data = next(iter(received.items()))
        safe_name = Path(name).name
        if Path(safe_name).suffix.lower() not in allowed or len(data) > 100 * 1024**2:
            raise ValueError('Formato não aceito ou arquivo maior que 100 MB.')
        destination = upload_dir / safe_name
        if destination.exists():
            destination = upload_dir / (uuid.uuid4().hex + Path(safe_name).suffix)
        destination.write_bytes(data)
        return str(destination)

    video_path = upload_one('1. Escolha o vídeo:', {'.mp4', '.mov', '.webm', '.avi'})
    reference_path = upload_one('2. Escolha a imagem do personagem/objeto:', {'.png', '.jpg', '.jpeg', '.webp'})
    """),
    cell("markdown", """
    ## Escolher a transformação

    - **motion**: seguir os movimentos corporais; use uma pessoa bem visível. Não transfere expressão facial.
    - **swap**: regenerar a região indicada usando a imagem do substituto. Comece com objeto parado.

    Região = esquerda, topo, direita, base, entre 0 e 1, referida à prévia redimensionada.
    Deixe a região final vazia para manter uma caixa fixa. Duas caixas produzem
    deslocamento linear, não rastreamento inteligente ou tratamento especial de oclusões.
    """),
    cell("code", """
    mode = 'motion' #@param ['motion', 'swap']
    prompt = 'A person wearing a red jacket raises their arms.' #@param {type:'string'}
    frames = 17 #@param [17, 33, 49, 81] {type:'raw'}
    steps = 25 #@param {type:'slider', min:10, max:50, step:1}
    seed = 42 #@param {type:'integer'}
    start_box = '0.25,0.2,0.75,0.85' #@param {type:'string'}
    end_box = '' #@param {type:'string'}
    size = (256, 144) # Rascunho de baixo consumo; a qualidade será limitada.
    """),
    cell("code", """
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from media import read_clip, box_masks, mask_preview
    from IPython.display import display
    clip = read_clip(video_path, size, frames)
    if mode == 'swap':
        masks = box_masks(size, len(clip), start_box, end_box)
        print('Primeiro e último quadro: a área vermelha será editada. Ajuste a região antes de gerar.')
        display(mask_preview(clip[0], masks[0]), mask_preview(clip[-1], masks[-1]))
    else:
        print('Primeiro quadro do trecho de movimento:')
        display(clip[0])
    """),
    cell("markdown", """
    ## Gerar e baixar

    O primeiro uso baixa o VACE e, no modo movimento, o detector de pose.
    Este passo executa o modelo real. Uma falha não será substituída por vídeo de exemplo.
    Se faltar memória, reduza para 17 quadros. Não há tempo de processamento garantido.
    Baixe seus resultados antes de encerrar a sessão; os arquivos do Colab são temporários.
    """),
    cell("code", """
    from engine import generate
    from IPython.display import Video, display
    result, manifest = generate(video_path, reference_path, mode, prompt, size, frames,
                                steps, seed, start_box, end_box, report=print)
    display(Video(result, embed=True))
    print('Registro:', manifest)
    files.download(result)
    files.download(manifest)
    """),
]
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"}}
target = ROOT / "notebooks" / "Genjutsu_Pessoal_Colab.ipynb"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(target)
