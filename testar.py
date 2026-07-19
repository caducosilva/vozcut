"""Bateria de testes do perfil de voz. Rode sempre que retreinar.

Uso:
    python testar.py <video_com_sua_voz>

Gera casos de teste na hora (sua voz limpa, sua voz com ruido por cima,
vozes sinteticas ineditas e ruido puro) e confere se o sistema aceita o
que deve aceitar e rejeita o que deve rejeitar.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from vozcut_lib import (
    assinatura,
    SAMPLE_RATE,
    extrair_audio, carregar_wav, detectar_fala, embeddings_por_janela,
    carregar_perfil, e_voz_do_dono,
)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", *args],
                   check=True, capture_output=True, stdin=subprocess.DEVNULL)


def tts(destino, texto, rate=1):
    """Gera uma fala sintetica inedita (nao usada no treino do anti-perfil)."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{destino}'); $s.Speak('{texto}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   check=True, capture_output=True)


def taxa_aceitacao(arquivo, perfil, limiar, negativos):
    """Percentual de janelas de fala aceitas como voz do dono (None = sem fala)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "a.wav")
        extrair_audio(str(arquivo), wav_path)
        wav = carregar_wav(wav_path)
    segmentos = detectar_fala(wav)
    if not segmentos:
        return None
    aceitas = total = 0
    for seg in segmentos:
        embs, _ = embeddings_por_janela(wav, seg)
        for e in embs:
            total += 1
            aceitas += e_voz_do_dono(e, perfil, limiar, negativos)[0]
    return 100.0 * aceitas / total


def main():
    assinatura()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video_dono = sys.argv[1]
    perfil, limiar, negativos = carregar_perfil()
    n_neg = 0 if negativos is None else len(negativos)
    print(f"Perfil carregado | limiar {limiar:.3f} | anti-perfil com {n_neg} exemplos\n")

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(tmp)
        print("Preparando casos de teste...")

        sua_voz = pasta / "sua_voz.wav"
        ffmpeg("-ss", "5", "-t", "40", "-i", video_dono,
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(sua_voz))

        sua_voz_ruido = pasta / "sua_voz_com_ruido.wav"
        ffmpeg("-i", str(sua_voz),
               "-f", "lavfi", "-i", f"anoisesrc=color=pink:duration=40:amplitude=0.08",
               "-filter_complex", "amix=inputs=2:duration=first:normalize=0",
               "-ar", str(SAMPLE_RATE), str(sua_voz_ruido))

        voz_outra_1 = pasta / "voz_sintetica_nova_1.wav"
        tts(voz_outra_1, "Este texto nunca foi usado no treinamento do anti perfil, "
            "e uma fala completamente inedita para testar a rejeicao.", rate=1)

        voz_outra_2 = pasta / "voz_sintetica_nova_2.wav"
        tts(voz_outra_2, "Mais uma locucao diferente com outro ritmo de fala "
            "para conferir se o sistema continua rejeitando.", rate=-2)

        ruido = pasta / "ruido_puro.wav"
        ffmpeg("-f", "lavfi", "-i", "anoisesrc=color=white:duration=10:amplitude=0.5",
               "-ar", str(SAMPLE_RATE), str(ruido))

        # (nome, arquivo, deve_aceitar_maioria)
        casos = [
            ("Sua voz limpa", sua_voz, True),
            ("Sua voz com ruido por cima", sua_voz_ruido, True),
            ("Voz sintetica inedita 1", voz_outra_1, False),
            ("Voz sintetica inedita 2", voz_outra_2, False),
            ("Ruido puro (sem fala)", ruido, False),
        ]

        print(f"\n{'CASO':35} {'ACEITO':>8}  ESPERADO  RESULTADO")
        print("-" * 68)
        falhas = 0
        for nome, arquivo, deve in casos:
            taxa = taxa_aceitacao(arquivo, perfil, limiar, negativos)
            if taxa is None:
                ok = not deve
                mostrado = "sem fala"
            else:
                ok = (taxa >= 60) if deve else (taxa <= 10)
                mostrado = f"{taxa:.0f}%"
            falhas += not ok
            esperado = "aceitar" if deve else "rejeitar"
            print(f"{nome:35} {mostrado:>8}  {esperado:8}  {'PASSOU' if ok else 'FALHOU'}")

        print("-" * 68)
        if falhas:
            sys.exit(f"\n{falhas} teste(s) FALHARAM. Reveja o treino ou o limiar.")
        print("\nTodos os testes passaram. O perfil esta afiado.")


if __name__ == "__main__":
    main()
