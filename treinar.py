"""Treina (enrola) o perfil de voz do Carlos a partir de um ou mais videos/audios.

Uso:
    python treinar.py video1.mp4 [video2.mp4 ...]

Quanto mais videos com a sua voz voce passar, mais robusto o perfil fica.
Rodar de novo com novos videos ACUMULA com o que ja foi aprendido.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from vozcut_lib import (
    assinatura,
    PERFIL_PATH, SAMPLE_RATE,
    extrair_audio, carregar_wav, detectar_fala, embeddings_por_janela,
)


def coletar_embeddings(video_path: str):
    print(f"[1/3] Extraindo audio de {Path(video_path).name}...")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extrair_audio(video_path, wav_path)
        wav = carregar_wav(wav_path)

    print("[2/3] Detectando trechos com fala (Silero VAD)...")
    segmentos = detectar_fala(wav)
    total_fala = sum(s["end"] - s["start"] for s in segmentos) / SAMPLE_RATE
    print(f"      {len(segmentos)} trechos de fala, {total_fala:.1f}s no total")
    if total_fala < 10:
        print("      AVISO: menos de 10s de fala. O perfil pode ficar fraco.")

    print("[3/3] Calculando assinaturas de voz (ECAPA)...")
    embs = []
    for seg in segmentos:
        e, _ = embeddings_por_janela(wav, seg)
        embs.extend(e)
    return np.array(embs)


def filtrar_voz_dominante(embs: np.ndarray):
    """Remove janelas que nao parecem ser da voz dominante do video.

    Protege o perfil caso o video de treino tenha outros sons/vozes:
    calcula o centro, descarta o que esta longe dele e repete.
    """
    manter = np.ones(len(embs), dtype=bool)
    for _ in range(3):
        centro = embs[manter].mean(axis=0)
        centro /= np.linalg.norm(centro) + 1e-9
        sims = embs @ centro
        corte = max(0.25, sims[manter].mean() - 2 * sims[manter].std())
        novo = sims >= corte
        if novo.sum() == manter.sum():
            break
        manter = novo
    return manter


def main():
    assinatura()
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    embs = np.concatenate([coletar_embeddings(v) for v in sys.argv[1:]])

    if PERFIL_PATH.exists():
        antigos = np.load(PERFIL_PATH)
        if "embs" in antigos:
            print(f"Acumulando com {len(antigos['embs'])} janelas do treino anterior.")
            embs = np.concatenate([antigos["embs"], embs])

    manter = filtrar_voz_dominante(embs)
    descartadas = int((~manter).sum())
    if descartadas:
        print(f"Descartadas {descartadas} janelas que nao batem com a voz dominante "
              "(provavel ruido ou outra voz).")

    perfil = embs[manter].mean(axis=0)
    perfil /= np.linalg.norm(perfil) + 1e-9

    # Calibra um limiar: mediana das similaridades das proprias janelas menos folga
    sims = embs[manter] @ perfil
    limiar = float(max(0.25, np.percentile(sims, 5) - 0.10))

    np.savez(PERFIL_PATH, perfil=perfil, embs=embs[manter], limiar_sugerido=limiar)
    print(f"\nPerfil salvo em {PERFIL_PATH}")
    print(f"Janelas usadas: {int(manter.sum())} | similaridade media: {sims.mean():.3f}")
    print(f"Limiar sugerido para o corte: {limiar:.3f}")


if __name__ == "__main__":
    main()
