"""Interface web local do VozCut. Rode e abra http://127.0.0.1:7860

Uso:
    python interface.py

Tudo roda na sua maquina; a pagina so funciona localmente.
"""
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

PROJETO = Path(__file__).parent
PYTHON = sys.executable
PERFIL = PROJETO / "perfil_voz.npz"
SAIDA = PROJETO / "edite_videos"

app = Flask(__name__)

SCRIPTS = {
    "cortar": "cortar.py",
    "treinar": "treinar.py",
    "verificar": "verificar.py",
}

job = {"ativo": False, "acao": None, "linhas": []}
trava = threading.Lock()


def estado_perfil():
    if not PERFIL.exists():
        return {"treinado": False, "janelas": 0, "negativos": 0, "limiar": None}
    d = np.load(PERFIL)
    return {
        "treinado": True,
        "janelas": int(len(d["embs"])) if "embs" in d else 0,
        "negativos": int(len(d["negativos"])) if "negativos" in d else 0,
        "limiar": round(float(d["limiar_sugerido"]), 3),
    }


def videos_prontos():
    if not SAIDA.exists():
        return []
    arquivos = sorted(SAIDA.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"nome": p.name, "mb": round(p.stat().st_size / 1e6, 1)} for p in arquivos[:12]]


def escolher_arquivos():
    import tkinter as tk
    from tkinter import filedialog
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    arquivos = filedialog.askopenfilenames(
        title="Escolha o(s) arquivo(s)",
        filetypes=[("Videos e audios", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wav *.mp3"),
                   ("Todos", "*.*")],
    )
    raiz.destroy()
    return list(arquivos)


def rodar(acao, caminhos):
    proc = subprocess.Popen(
        [PYTHON, "-u", str(PROJETO / SCRIPTS[acao]), *caminhos],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, cwd=str(PROJETO),
    )
    for linha in proc.stdout:
        linha = linha.rstrip()
        if linha and "UserWarning" not in linha and "warnings.warn" not in linha:
            job["linhas"].append(linha)
    proc.wait()
    job["linhas"].append("")
    job["linhas"].append(f"[{acao} finalizado com codigo {proc.returncode}]")
    job["ativo"] = False


@app.get("/")
def pagina():
    return render_template("index.html", perfil=estado_perfil(), prontos=videos_prontos())


@app.post("/executar")
def executar():
    dados = request.get_json(force=True)
    acao = dados.get("acao")
    if acao not in SCRIPTS:
        return jsonify(erro="acao invalida"), 400
    with trava:
        if job["ativo"]:
            return jsonify(erro="ja existe um processamento em andamento"), 409
        caminhos = [c.strip().strip('"') for c in dados.get("caminhos", []) if c.strip()]
        if not caminhos:
            caminhos = escolher_arquivos()
        if not caminhos:
            return jsonify(erro="nenhum arquivo escolhido"), 400
        job.update(ativo=True, acao=acao, linhas=[f"[{acao}] {len(caminhos)} arquivo(s)"])
        threading.Thread(target=rodar, args=(acao, caminhos), daemon=True).start()
    return jsonify(ok=True)


@app.get("/status")
def status():
    return jsonify(ativo=job["ativo"], acao=job["acao"], linhas=job["linhas"],
                   perfil=estado_perfil(), prontos=videos_prontos())


if __name__ == "__main__":
    print("VozCut rodando em http://127.0.0.1:7860 (Ctrl+C para sair)")
    app.run(host="127.0.0.1", port=7860, debug=False)
