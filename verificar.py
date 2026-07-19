"""Verifica o quanto um audio/video parece com o perfil de voz treinado.

Uso:
    python verificar.py arquivo1.mp4 [arquivo2.wav ...]

Para cada arquivo, mostra a similaridade de cada janela de fala com o seu
perfil e um veredito geral. Use para testar o perfil: rode com um video seu
(deve dar alto) e com um audio de outra pessoa (deve dar baixo).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from vozcut_lib import (
    assinatura,
    SAMPLE_RATE,
    extrair_audio, carregar_wav, detectar_fala, embeddings_por_janela,
    carregar_perfil, classificar_segmento,
)


def verificar(arquivo, perfil, limiar, negativos):
    print(f"\n===== {Path(arquivo).name} =====")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extrair_audio(arquivo, wav_path)
        wav = carregar_wav(wav_path)

    segmentos = detectar_fala(wav)
    if not segmentos:
        print("Nenhuma fala detectada.")
        return

    sims = []
    for seg in segmentos:
        embs, janelas = embeddings_por_janela(wav, seg)
        ok, sims_p, sims_n = classificar_segmento(embs, perfil, limiar, negativos)
        for aceito, sim_p, sim_n, (ini, fim) in zip(ok, sims_p, sims_n, janelas):
            sims.append((ini / SAMPLE_RATE, fim / SAMPLE_RATE, sim_p, sim_n, aceito))

    aceitas = sum(1 for s in sims if s[4])
    valores = [s[2] for s in sims]
    print(f"Janelas de fala: {len(sims)} | limiar: {limiar:.3f}")
    print(f"Similaridade com voce: media {np.mean(valores):.3f} | "
          f"minima {min(valores):.3f} | maxima {max(valores):.3f}")
    print(f"Reconhecidas como SUA voz: {aceitas}/{len(sims)} "
          f"({100 * aceitas / len(sims):.0f}%)")
    if "--detalhe" in sys.argv:
        for ini, fim, sim_p, sim_n, aceito in sims:
            marca = "VOCE " if aceito else "outro"
            extra = f" (anti-perfil {sim_n:.3f})" if sim_n >= 0 else ""
            print(f"  {ini:7.2f}s -> {fim:7.2f}s  sim={sim_p:.3f}{extra}  [{marca}]")


def main():
    assinatura()
    arquivos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not arquivos:
        sys.exit(__doc__)
    perfil, limiar, negativos = carregar_perfil()
    if negativos is not None:
        print(f"Anti-perfil ativo: {len(negativos)} exemplos do que nao e a sua voz.")
    for a in arquivos:
        verificar(a, perfil, limiar, negativos)


if __name__ == "__main__":
    main()
