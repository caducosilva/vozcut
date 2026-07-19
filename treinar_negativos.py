"""Ensina ao sistema exemplos do que NAO e a sua voz (anti-perfil).

Gera automaticamente vozes sinteticas do Windows e ruidos (branco, rosa,
marrom, zumbido de rede eletrica), calcula as assinaturas de cada um e salva
junto ao perfil. No corte, um trecho so e aceito se parecer com a SUA voz
E nao parecer mais com algum desses exemplos negativos.

Uso:
    python treinar_negativos.py [audio_de_outra_pessoa.mp4 ...]

Sem argumentos, usa so os exemplos sinteticos. Passar gravacoes reais de
outras vozes (familia, amigos, TV) deixa o anti-perfil ainda mais forte.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vozcut_lib import (
    assinatura,
    PERFIL_PATH, SAMPLE_RATE,
    extrair_audio, carregar_wav, detectar_fala, embeddings_por_janela, embedding,
)

TEXTOS = [
    "Ola, esta e uma voz de teste para o sistema aprender o que nao e a voz do dono.",
    "The quick brown fox jumps over the lazy dog while the system listens carefully.",
    "Um dois tres quatro cinco seis sete oito nove dez, testando a calibracao de voz.",
]

RUIDOS = {
    "ruido_branco": "anoisesrc=color=white:duration=8:amplitude=0.5",
    "ruido_rosa": "anoisesrc=color=pink:duration=8:amplitude=0.5",
    "ruido_marrom": "anoisesrc=color=brown:duration=8:amplitude=0.7",
    "zumbido_60hz": "sine=frequency=60:duration=8,volume=0.5",
    "apito_1khz": "sine=frequency=1000:duration=8,volume=0.3",
}


def gerar_tts(pasta: Path):
    """Gera falas com todas as vozes instaladas no Windows, em varios ritmos."""
    arquivos = []
    script = pasta / "tts.ps1"
    linhas = ["Add-Type -AssemblyName System.Speech",
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer"]
    linhas.append("$vozes = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }")
    for i, texto in enumerate(TEXTOS):
        for rate in (-3, 0, 4):
            nome = f"tts_{i}_{rate}.wav"
            arquivos.append(pasta / nome)
            linhas.append("foreach ($v in $vozes) { $s.SelectVoice($v); "
                          f"$s.Rate = {rate}; "
                          f"$s.SetOutputToWaveFile('{pasta / nome}'.Replace('.wav', \"_$v.wav\")); "
                          f"$s.Speak('{texto}') }}")
    linhas.append("$s.Dispose()")
    script.write_text("\n".join(linhas), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(script)], check=True, capture_output=True)
    return sorted(pasta.glob("tts_*.wav"))


def gerar_ruidos(pasta: Path):
    arquivos = []
    for nome, filtro in RUIDOS.items():
        destino = pasta / f"{nome}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", filtro,
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(destino)],
            check=True, capture_output=True, stdin=subprocess.DEVNULL)
        arquivos.append(destino)
    return arquivos


def embeddings_de_fala(caminho) -> list:
    """Assinaturas das partes faladas de um arquivo (usa VAD)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extrair_audio(str(caminho), wav_path)
        wav = carregar_wav(wav_path)
    embs = []
    for seg in detectar_fala(wav):
        e, _ = embeddings_por_janela(wav, seg)
        embs.extend(e)
    return embs


def embeddings_de_ruido(caminho) -> list:
    """Assinaturas de janelas fixas (ruido nao passa no VAD, entao corta na marra)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        extrair_audio(str(caminho), wav_path)
        wav = carregar_wav(wav_path)
    passo = int(2.5 * SAMPLE_RATE)
    return [embedding(wav[i:i + passo])
            for i in range(0, max(1, len(wav) - passo), passo)]


def main():
    assinatura()
    if not PERFIL_PATH.exists():
        sys.exit("Treine primeiro o seu perfil: python treinar.py <video>")

    negativos = []
    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        print("[1/3] Gerando vozes sinteticas (TTS do Windows)...")
        for arq in gerar_tts(pasta):
            negativos.extend(embeddings_de_fala(arq))
        print(f"      {len(negativos)} assinaturas de vozes sinteticas")

        print("[2/3] Gerando ruidos (branco, rosa, marrom, zumbidos)...")
        antes = len(negativos)
        for arq in gerar_ruidos(pasta):
            negativos.extend(embeddings_de_ruido(arq))
        print(f"      {len(negativos) - antes} assinaturas de ruido")

    print("[3/3] Processando arquivos extras (vozes reais de outras pessoas)...")
    for extra in sys.argv[1:]:
        embs = embeddings_de_fala(extra)
        negativos.extend(embs)
        print(f"      {Path(extra).name}: {len(embs)} assinaturas")

    negativos = np.array(negativos)

    dados = dict(np.load(PERFIL_PATH))
    perfil = dados["perfil"]

    # Seguranca: se algum negativo parecer DEMAIS com o dono, e descartado
    # (protege contra passar sem querer um audio do proprio dono como negativo)
    sims = negativos @ perfil
    suspeitos = sims > 0.5
    if suspeitos.any():
        print(f"AVISO: {int(suspeitos.sum())} assinaturas negativas parecem com a SUA voz "
              "e foram descartadas por seguranca.")
        negativos = negativos[~suspeitos]

    if "negativos" in dados:
        print(f"Acumulando com {len(dados['negativos'])} negativos anteriores.")
        negativos = np.concatenate([dados["negativos"], negativos])

    dados["negativos"] = negativos
    np.savez(PERFIL_PATH, **dados)
    print(f"\nAnti-perfil salvo: {len(negativos)} assinaturas do que NAO e a sua voz.")
    print(f"Similaridade maxima delas com voce: {float((negativos @ perfil).max()):.3f} "
          "(quanto menor, melhor)")


if __name__ == "__main__":
    main()
