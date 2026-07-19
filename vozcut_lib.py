"""Funcoes compartilhadas do VozCut: extracao de audio, VAD e embeddings de voz."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000
PROJETO = Path(__file__).parent
PERFIL_PATH = PROJETO / "perfil_voz.npz"

ASSINATURA = "VozCut · criado por caducosilva · contato: abobicarlo@gmail.com"
PIX_DOACOES = "doacoes via PIX (chave aleatoria): f74458dc-2a36-49bd-9250-1cef4365ebb8"


def assinatura():
    """Imprime a marca do autor no inicio dos scripts."""
    print(ASSINATURA)
    print(PIX_DOACOES)
    print("-" * len(ASSINATURA))


def extrair_audio(video_path: str, wav_path: str):
    """Extrai o audio do video em WAV 16 kHz mono (formato que os modelos esperam)."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)


def carregar_wav(wav_path: str) -> torch.Tensor:
    import soundfile as sf
    dados, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"esperava {SAMPLE_RATE} Hz, veio {sr}"
    return torch.from_numpy(dados)


def detectar_fala(wav: torch.Tensor):
    """Roda o Silero VAD e devolve lista de dicts {start, end} em amostras."""
    from silero_vad import load_silero_vad, get_speech_timestamps
    modelo = load_silero_vad()
    return get_speech_timestamps(
        wav, modelo,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )


_classificador = None


def _get_classificador():
    global _classificador
    if _classificador is None:
        # No Windows, symlink exige privilegio de admin; forca copia de arquivos
        import speechbrain.utils.fetching as _sbfetch
        _link_original = _sbfetch.link_with_strategy
        _sbfetch.link_with_strategy = lambda src, dst, estrategia: _link_original(
            src, dst, _sbfetch.LocalStrategy.COPY
        )
        from speechbrain.inference.speaker import EncoderClassifier
        _classificador = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(PROJETO / "modelos" / "ecapa"),
            run_opts={"device": "cpu"},
        )
    return _classificador


def embedding(trecho: torch.Tensor) -> np.ndarray:
    """Embedding ECAPA (192 dims) de um trecho de audio. Normalizado em L2."""
    clf = _get_classificador()
    with torch.no_grad():
        emb = clf.encode_batch(trecho.unsqueeze(0)).squeeze().numpy()
    return emb / (np.linalg.norm(emb) + 1e-9)


def embeddings_por_janela(wav: torch.Tensor, seg, janela_s=2.5):
    """Divide um segmento de fala em janelas e devolve (embeddings, janelas em amostras).

    Janelas menores permitem separar, dentro de um mesmo segmento de fala,
    as partes onde o Carlos fala das partes onde outra pessoa/som aparece.
    """
    ini, fim = seg["start"], seg["end"]
    passo = int(janela_s * SAMPLE_RATE)
    minimo = int(0.6 * SAMPLE_RATE)  # ECAPA fica instavel com menos que isso
    janelas, embs = [], []
    pos = ini
    while pos < fim:
        j_fim = min(pos + passo, fim)
        if j_fim - pos < minimo:
            if janelas:
                janelas[-1] = (janelas[-1][0], j_fim)
            else:
                j_ini = max(ini, j_fim - minimo)
                janelas.append((j_ini, j_fim))
                embs.append(embedding(wav[j_ini:j_fim]))
            break
        janelas.append((pos, j_fim))
        embs.append(embedding(wav[pos:j_fim]))
        pos = j_fim
    return embs, janelas


# Um trecho so e aceito se a semelhanca com o dono superar a semelhanca com o
# melhor exemplo negativo por pelo menos esta margem
MARGEM_NEGATIVA = 0.05


def carregar_perfil():
    """Devolve (perfil, limiar, negativos). negativos e None se o anti-perfil nao foi treinado."""
    if not PERFIL_PATH.exists():
        sys.exit(f"Perfil de voz nao encontrado em {PERFIL_PATH}. Rode primeiro: python treinar.py <video>")
    d = np.load(PERFIL_PATH)
    negativos = d["negativos"] if "negativos" in d and len(d["negativos"]) else None
    return d["perfil"], float(d["limiar_sugerido"]), negativos


def e_voz_do_dono(emb: np.ndarray, perfil, limiar, negativos):
    """Criterio duplo: (aceito, similaridade_com_dono, similaridade_com_negativos)."""
    sim_p = float(np.dot(emb, perfil))
    sim_n = float((negativos @ emb).max()) if negativos is not None else -1.0
    aceito = sim_p >= limiar and sim_p >= sim_n + MARGEM_NEGATIVA
    return aceito, sim_p, sim_n
