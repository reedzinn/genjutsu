import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from engine import generate
from media import box_masks, mask_preview, read_clip

SIZES = {"Rascunho — 256 × 144": (256, 144), "384 × 224": (384, 224),
         "480 × 272": (480, 272)}


def preview(video, resolution, count, start, end):
    if not video:
        raise gr.Error("Envie um vídeo primeiro.")
    try:
        size = SIZES[resolution]
        frames = read_clip(video, size, int(count))
        masks = box_masks(size, len(frames), start, end)
        return mask_preview(frames[0], masks[0]), mask_preview(frames[-1], masks[-1])
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def run(video, reference, mode, prompt, resolution, count, steps, seed, start, end,
        progress=gr.Progress()):
    if not video or not reference:
        raise gr.Error("Envie o vídeo e a imagem de referência.")
    try:
        return generate(video, reference, mode, prompt, SIZES[resolution], int(count),
                        int(steps), int(seed), start, end,
                        report=lambda message: progress(None, desc=message))
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_app():
    with gr.Blocks(title="Genjutsu — laboratório pessoal") as demo:
        gr.Markdown("# Laboratório pessoal de vídeo\n"
                    "Modelo aberto VACE, sem API paga. **Protótipo experimental: a geração ainda "
                    "precisa ser validada em GPU.** Requer NVIDIA/CUDA; não há suporte validado para RX 580. "
                    "Comece com 17 quadros. A primeira execução baixa vários GB de modelos.")
        with gr.Row():
            video = gr.Video(label="Vídeo de movimento ou cena", sources=["upload"])
            reference = gr.Image(label="Imagem do personagem ou objeto substituto", type="filepath", sources=["upload"])
        mode = gr.Radio([("Transferir movimento corporal", "motion"),
                         ("Trocar objeto na região marcada", "swap")], value="motion", label="Função")
        prompt = gr.Textbox(label="Descreva o resultado desejado (preferencialmente em inglês)",
                            placeholder="A person wearing a red jacket raises their arms.")
        with gr.Row():
            resolution = gr.Dropdown(list(SIZES), value="Rascunho — 256 × 144", label="Resolução")
            count = gr.Dropdown([17, 33, 49, 81], value=17, label="Quadros: aproximadamente 1, 2, 3 ou 5 segundos")
        with gr.Accordion("Região para troca de objeto", open=False):
            gr.Markdown("Use esquerda, topo, direita, base entre 0 e 1. Por exemplo, "
                        "0.25,0.2,0.75,0.85. A área vermelha será editada. "
                        "Os números se referem à prévia redimensionada. Não há rastreamento automático: "
                        "se indicar duas regiões, o retângulo se move em linha reta entre elas.")
            start = gr.Textbox(value="0.25,0.2,0.75,0.85", label="Região no início")
            end = gr.Textbox(value="", label="Região no fim (vazio = região fixa)")
            preview_button = gr.Button("Conferir região antes de gerar")
            with gr.Row():
                first = gr.Image(label="Primeiro quadro", interactive=False)
                last = gr.Image(label="Último quadro do trecho", interactive=False)
            preview_button.click(preview, [video, resolution, count, start, end], [first, last])
        with gr.Accordion("Ajustes", open=False):
            steps = gr.Slider(10, 50, value=25, step=1, label="Etapas de geração")
            seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32-1, label="Semente")
        gr.Markdown("Limites: uma pessoa no modo movimento; só movimento corporal, sem transferência facial. "
                    "A troca de objeto depende da região fornecida. Resultados podem deformar ou alterar "
                    "o fundo. Saída sem áudio. Nenhuma promessa de qualidade equivalente ao Genjutsu.")
        submit = gr.Button("Gerar vídeo", variant="primary")
        result = gr.Video(label="Resultado gerado", interactive=False)
        manifest = gr.File(label="Registro da execução")
        submit.click(run, [video, reference, mode, prompt, resolution, count, steps, seed, start, end],
                     [result, manifest], concurrency_limit=1, concurrency_id="generation")
    return demo.queue(max_size=2)


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="100mb")
