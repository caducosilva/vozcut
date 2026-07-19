"""Corta videos mantendo somente os trechos onde o CARLOS esta falando.

Silencio, ruido de fundo e vozes de outras pessoas sao removidos.
Os videos originais NAO sao alterados: as versoes cortadas sao salvas
na pasta 'edite_videos', dentro da pasta da automacao.

Uso:
    python cortar.py                                  abre janela para escolher os videos
    python cortar.py video1.mp4 [video2.mp4 ...]      corta os videos informados
    Opcoes: [--limiar 0.30] [--folga 0.25] [--relatorio]

    --limiar    similaridade minima com o perfil de voz (padrao: o calibrado no treino).
                Aumente se estiver deixando passar sons errados; diminua se estiver
                cortando a sua propria fala.
    --folga     segundos mantidos antes/depois de cada fala (padrao 0.25)
    --relatorio so mostra o que seria cortado, sem gerar video
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from vozcut_lib import (
    assinatura,
    PROJETO, SAMPLE_RATE,
    extrair_audio, carregar_wav, detectar_fala, embeddings_por_janela,
    carregar_perfil, e_voz_do_dono,
)

PASTA_SAIDA = PROJETO / "edite_videos"


def escolher_videos():
    """Abre a janela do Windows para escolher um ou mais videos."""
    import tkinter as tk
    from tkinter import filedialog
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    arquivos = filedialog.askopenfilenames(
        title="Escolha o(s) video(s) para cortar os momentos mudos",
        filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("Todos", "*.*")],
    )
    raiz.destroy()
    return list(arquivos)


def caminho_saida(video: str) -> str:
    """Nome do arquivo cortado em edite_videos, sem sobrescrever os anteriores."""
    PASTA_SAIDA.mkdir(exist_ok=True)
    destino = PASTA_SAIDA / (Path(video).stem + ".mp4")
    n = 2
    while destino.exists():
        destino = PASTA_SAIDA / f"{Path(video).stem}_{n}.mp4"
        n += 1
    return str(destino)


def classificar_janelas(wav, segmentos, perfil, limiar, negativos):
    """Devolve lista de (inicio_s, fim_s, similaridade) das janelas aceitas."""
    aceitas, recusadas = [], []
    for seg in segmentos:
        embs, janelas = embeddings_por_janela(wav, seg)
        for e, (ini, fim) in zip(embs, janelas):
            aceito, sim, _ = e_voz_do_dono(e, perfil, limiar, negativos)
            alvo = aceitas if aceito else recusadas
            alvo.append((ini / SAMPLE_RATE, fim / SAMPLE_RATE, sim))
    return aceitas, recusadas


def fundir(trechos, folga, gap_max=0.5, dur_total=None):
    """Aplica folga, funde trechos proximos e devolve [(ini, fim), ...]."""
    if not trechos:
        return []
    trechos = sorted((max(0.0, i - folga), f + folga) for i, f, _ in trechos)
    saida = [list(trechos[0])]
    for ini, fim in trechos[1:]:
        if ini - saida[-1][1] <= gap_max:
            saida[-1][1] = max(saida[-1][1], fim)
        else:
            saida.append([ini, fim])
    if dur_total:
        for t in saida:
            t[1] = min(t[1], dur_total)
    return [(i, f) for i, f in saida if f - i > 0.05]


def montar_video(video_path, trechos, saida):
    """Corta e concatena os trechos com ffmpeg (frame-accurate, re-encode)."""
    partes_v, partes_a, filtros = [], [], []
    for n, (ini, fim) in enumerate(trechos):
        filtros.append(
            f"[0:v]trim=start={ini:.3f}:end={fim:.3f},setpts=PTS-STARTPTS[v{n}];"
            f"[0:a]atrim=start={ini:.3f}:end={fim:.3f},asetpts=PTS-STARTPTS[a{n}]"
        )
        partes_v.append(f"[v{n}]")
        partes_a.append(f"[a{n}]")
    n = len(trechos)
    filtros.append("".join(f"{v}{a}" for v, a in zip(partes_v, partes_a))
                   + f"concat=n={n}:v=1:a=1[vf][af]")
    base = [
        "ffmpeg", "-nostdin", "-y", "-i", video_path,
        "-filter_complex", ";".join(filtros),
        "-map", "[vf]", "-map", "[af]",
        "-c:a", "aac", "-b:a", "192k",
    ]

    def rodar(cmd):
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
        return proc.returncode == 0

    # Tenta o encoder de hardware (Intel Quick Sync); se falhar, cai pro software
    qsv = base + ["-c:v", "h264_qsv", "-global_quality", "20", "-preset", "fast", saida]
    if not rodar(qsv):
        print("Encoder de hardware falhou, usando libx264 (mais lento)...")
        x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", saida]
        if not rodar(x264):
            sys.exit("ffmpeg falhou nos dois encoders. Veja o erro acima.")


def processar(video, perfil, limiar, negativos, folga, so_relatorio):
    print(f"\n===== {Path(video).name} =====")
    print("[1/4] Extraindo audio...")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extrair_audio(video, wav_path)
        wav = carregar_wav(wav_path)
    dur_total = len(wav) / SAMPLE_RATE

    print("[2/4] Detectando fala (Silero VAD)...")
    segmentos = detectar_fala(wav)

    print("[3/4] Verificando quem esta falando (perfil de voz)...")
    aceitas, recusadas = classificar_janelas(wav, segmentos, perfil, limiar, negativos)
    trechos = fundir(aceitas, folga, dur_total=dur_total)

    mantido = sum(f - i for i, f in trechos)
    print(f"Video original: {dur_total:.1f}s")
    print(f"Mantido (sua voz): {mantido:.1f}s em {len(trechos)} trechos")
    print(f"Removido: {dur_total - mantido:.1f}s")
    if recusadas:
        print(f"Janelas com som mas que NAO bateram com sua voz: {len(recusadas)} "
              f"(similaridade media {np.mean([s for _, _, s in recusadas]):.3f})")

    if so_relatorio:
        for i, f in trechos:
            print(f"  mantem {i:7.2f}s -> {f:7.2f}s")
        return None
    if not trechos:
        print("AVISO: nenhum trecho com a sua voz. Pulado. Tente diminuir o --limiar.")
        return None

    saida = caminho_saida(video)
    print("[4/4] Montando video final (pode demorar se for 4K)...")
    montar_video(video, trechos, saida)
    print(f"Salvo em: {saida}")
    return saida


def main():
    assinatura()
    videos = [a for a in sys.argv[1:] if not a.startswith("--")
              and sys.argv[max(0, sys.argv.index(a) - 1)] not in ("--limiar", "--folga")]
    if not videos:
        videos = escolher_videos()
        if not videos:
            sys.exit("Nenhum video escolhido.")

    faltando = [v for v in videos if not Path(v).exists()]
    if faltando:
        sys.exit("Video(s) nao encontrado(s): " + ", ".join(faltando))

    perfil, limiar, negativos = carregar_perfil()
    if "--limiar" in sys.argv:
        limiar = float(sys.argv[sys.argv.index("--limiar") + 1])
    folga = 0.25
    if "--folga" in sys.argv:
        folga = float(sys.argv[sys.argv.index("--folga") + 1])
    so_relatorio = "--relatorio" in sys.argv

    print(f"Limiar de similaridade: {limiar:.3f}")
    print(f"{len(videos)} video(s) para processar. Saida em: {PASTA_SAIDA}")

    prontos = []
    for video in videos:
        resultado = processar(video, perfil, limiar, negativos, folga, so_relatorio)
        if resultado:
            prontos.append(resultado)

    if not so_relatorio:
        print(f"\n{'=' * 40}")
        print(f"Concluido! {len(prontos)} de {len(videos)} video(s) cortado(s) em:")
        print(f"  {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
