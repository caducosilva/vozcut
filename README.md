# VozCut

Editor automatico de cortes que reconhece **a voz do dono** e remove dos videos
tudo que nao for ele falando: silencio, ruido de fundo e vozes de outras pessoas.

Feito para criadores de conteudo que gravam sozinhos e perdem tempo cortando
momentos mudos na mao. Roda 100% na sua maquina, offline apos a primeira
execucao. Nenhum audio ou video sai do seu computador.

## Como funciona

1. **Silero VAD** encontra os trechos do audio que contem fala
2. **ECAPA-TDNN (SpeechBrain)** transforma cada trecho numa assinatura de voz de 192 dimensoes
3. O **perfil de voz** do dono e a media das assinaturas dos videos de treino
4. O **anti-perfil** guarda assinaturas do que NAO e o dono: vozes sinteticas em varios ritmos e ruidos (branco, rosa, marrom, zumbido de rede eletrica)
5. Um trecho so entra no video final se passar no **criterio duplo**: parecer com o dono acima do limiar calibrado E parecer mais com o dono do que com qualquer exemplo do anti-perfil
6. O **FFmpeg** corta e concatena os trechos aprovados com precisao de frame, usando encoder de hardware quando disponivel (Intel Quick Sync), com fallback para libx264

## Requisitos

- Windows 10/11 (o gerador de vozes do anti-perfil usa o TTS do Windows; o restante funciona em Linux/macOS)
- Python 3.10 ou superior
- FFmpeg no PATH (`winget install Gyan.FFmpeg`)

## Instalacao

```
git clone https://github.com/caducosilva/vozcut.git
cd vozcut
python -m venv venv
venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip install -r requirements.txt
```

Na primeira execucao os modelos abertos (~100 MB) sao baixados uma unica vez.

## Uso

### Interface web (recomendado)

De dois cliques em `INTERFACE.bat` (ou rode `venv\Scripts\python interface.py`)
e abra http://127.0.0.1:7860. De la voce treina o perfil, corta videos e
verifica arquivos acompanhando o progresso ao vivo.

### Atalhos

- `CORTAR.bat`: escolha um ou mais videos na janela que abre (ou arraste videos
  em cima do .bat). As versoes cortadas vao para a pasta `edite_videos`, com os
  originais intactos
- `TREINAR.bat`: arraste um video seu em cima para criar ou reforcar o perfil

### Linha de comando

```
venv\Scripts\python treinar.py meu_video.mp4        cria/reforca o perfil de voz
venv\Scripts\python treinar_negativos.py            treina o anti-perfil (vozes e ruidos)
venv\Scripts\python cortar.py video1.mp4 video2.mp4 corta os videos
venv\Scripts\python verificar.py arquivo.mp4        mede a similaridade com o perfil
venv\Scripts\python testar.py meu_video.mp4         bateria de testes completa
```

Opcoes do `cortar.py`:

| Opcao | Efeito |
|---|---|
| `--relatorio` | so mostra o que seria mantido/cortado, sem gerar video |
| `--limiar 0.5` | similaridade minima; diminua se cortar sua fala, aumente se passar ruido |
| `--folga 0.25` | segundos de respiro mantidos antes/depois de cada fala |

## Resultados de teste

Bateria do `testar.py` num perfil treinado com ~3 minutos de fala:

| Caso | Aceito | Esperado |
|---|---|---|
| Voz do dono limpa | 100% | aceitar |
| Voz do dono com ruido por cima | 100% | aceitar |
| Voz sintetica inedita 1 | 0% | rejeitar |
| Voz sintetica inedita 2 | 0% | rejeitar |
| Ruido puro | sem fala detectada | rejeitar |

## Dicas para um perfil forte

- Treine com 2 a 5 videos seus em ambientes diferentes; o treino acumula
- Gravou num ambiente novo (rua, sol, vento)? Treine com o primeiro video de la antes de cortar os demais
- Tem gravacao de outras pessoas? Passe para o anti-perfil: `python treinar_negativos.py voz_do_amigo.mp4`
- Ao ar livre, uma espuma de microfone (deadcat) melhora muito a deteccao

## Privacidade

O perfil de voz (`perfil_voz.npz`), os modelos baixados e os videos ficam fora
do repositorio (veja o `.gitignore`). Compartilhe o codigo a vontade; a sua voz
fica so com voce.

## Autor

Criado por **caducosilva**. Duvidas, ideias ou parcerias:
[abobicarlo@gmail.com](mailto:abobicarlo@gmail.com)

Gostou do projeto e quer apoiar? Doacoes via PIX (chave aleatoria):

```
f74458dc-2a36-49bd-9250-1cef4365ebb8
```

## Licenca

[MIT](LICENSE). Os modelos usados tem licencas proprias: Silero VAD (MIT) e
SpeechBrain ECAPA-TDNN (Apache 2.0).
