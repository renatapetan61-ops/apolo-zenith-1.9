"""Baixa os datasets multimodais do Apolo Zenith 1.9 de forma **verificada**.

Fluxo por dataset:

1. verifica espaço livre (mínimo de segurança);
2. baixa em blocos (arquivo ``.part`` intermediário);
3. confere o SHA-256;
4. conta as amostras de verdade;
5. grava ``dataset_manifest.json`` e ``logs/dataset_downloads.log``.

Uso::

    python -m scripts.download_datasets --datasets codigo_github_permissivo
    python -m scripts.download_datasets --modalidades imagens,audio
    python -m scripts.download_datasets --listar
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from datasets.catalog import CATALOGO, DatasetSpec, listar, obter  # noqa: E402
from datasets.downloader import (  # noqa: E402
    Downloader,
    baixar_repo_github,
    branches_candidatos,
    info_licenca_repo,
    listar_arquivos_repo,
    listar_videos_commons,
)
from datasets.manifest import DatasetManifest, EntradaManifesto  # noqa: E402
from resources.config import carregar_config_recursos  # noqa: E402


def _tamanho_dir(caminho: Path) -> int:
    total = 0
    for arquivo in caminho.rglob("*"):
        if arquivo.is_file():
            try:
                total += arquivo.stat().st_size
            except OSError:
                pass
    return total


def _entrada_base(spec: DatasetSpec) -> EntradaManifesto:
    return EntradaManifesto(
        nome=spec.nome,
        modalidade=spec.modalidade,
        origem=spec.origem,
        url_oficial=spec.url_oficial,
        licenca=spec.licenca,
        formato=spec.formato,
        observacoes=spec.notas,
    )


# --------------------------------------------------------------------------- #
# Preparadores por tipo
# --------------------------------------------------------------------------- #
def preparar_texto(spec: DatasetSpec, base: Path, manifesto: DatasetManifest, downloader: Downloader) -> EntradaManifesto:
    """Registra o corpus textual já existente (sem re-download)."""
    entrada = manifesto.entradas.get(spec.nome) or _entrada_base(spec)
    origem = RAIZ / "datasets_storage" / "raw"
    processado = RAIZ / "datasets_storage" / "processed"
    entrada.tamanho_bytes = _tamanho_dir(origem)
    entrada.tamanho_em_disco_bytes = _tamanho_dir(origem) + _tamanho_dir(processado)
    entrada.status_download = "concluido"
    entrada.status_preparacao = "concluido"
    # Contagem real de documentos processados.
    train_txt = processado / "train.txt"
    entrada.amostras = (
        len(train_txt.read_text(encoding="utf-8", errors="replace").split("\n\n"))
        if train_txt.exists()
        else None
    )
    if entrada.amostras is None:
        entrada.amostras = None
    entrada.sha256 = None
    entrada.observacoes = (
        spec.notas + " SHA-256 por arquivo disponível em datasets_storage/manifest.json."
    )
    manifesto.registrar(entrada)
    return entrada


def preparar_codigo(spec: DatasetSpec, base: Path, manifesto: DatasetManifest, downloader: Downloader) -> EntradaManifesto:
    """Baixa os arquivos de código dos repositórios listados."""
    entrada = manifesto.entradas.get(spec.nome) or _entrada_base(spec)
    entrada.status_download = "em_andamento"
    manifesto.registrar(entrada)

    destino_base = base / "raw"
    total_bytes = 0
    arquivos_baixados = 0
    subfontes: List[Dict[str, Any]] = []

    for repo in spec.repos:
        repo_dir = destino_base / repo.linguagem / repo.repo
        # Repositório já baixado? Não repete o trabalho.
        if repo_dir.exists():
            ja_prontos = sum(1 for p in repo_dir.rglob("*") if p.is_file())
            if ja_prontos >= repo.max_arquivos:
                arquivos_baixados += ja_prontos
                total_bytes += _tamanho_dir(repo_dir)
                subfontes.append(
                    {
                        "repo": repo.nome,
                        "url": repo.url_oficial,
                        "linguagem": repo.linguagem,
                        "licenca": repo.licenca_esperada,
                        "licenca_confirmada_api": False,
                        "modo": "reaproveitado",
                        "arquivos": ja_prontos,
                        "bytes": _tamanho_dir(repo_dir),
                    }
                )
                print(
                    f"  [codigo] {repo.nome:32s} {repo.linguagem:11s} {ja_prontos:3d} arquivos "
                    f"(reaproveitado)"
                )
                continue

        info = info_licenca_repo(repo.owner, repo.repo)
        branch = repo.branch or info.get("branch_padrao") or "main"
        licenca_real = info.get("licenca") or repo.licenca_esperada

        # 1) Tenta a API de árvore (rápida, mas com limite de requisições).
        try:
            arquivos = listar_arquivos_repo(
                repo.owner,
                repo.repo,
                branch,
                repo.incluir,
                repo.excluir,
                repo.max_arquivos,
                repo.max_bytes_por_arquivo,
            )
        except Exception:
            arquivos = []

        bytes_repo = 0
        ok_repo = 0
        if arquivos:
            repo_dir.mkdir(parents=True, exist_ok=True)
            for item in arquivos:
                destino = repo_dir / item["path"]
                resultado = downloader.baixar(
                    item["url"], destino, tamanho_esperado=None
                )
                if resultado.ok:
                    bytes_repo += resultado.bytes
                    ok_repo += 1
            modo = "api"
        else:
            # 2) Fallback: tarball do repositório (sem limite da API).
            resultado_repo = None
            for candidato in branches_candidatos(repo.owner, repo.repo, repo.branch):
                resultado_repo = baixar_repo_github(
                    repo.owner,
                    repo.repo,
                    candidato,
                    repo_dir,
                    repo.incluir,
                    repo.excluir,
                    repo.max_arquivos,
                    repo.max_bytes_por_arquivo,
                    downloader,
                )
                if resultado_repo.get("arquivos"):
                    break
            ok_repo = int(resultado_repo.get("arquivos", 0))
            bytes_repo = int(resultado_repo.get("bytes", 0))
            branch = resultado_repo.get("branch", branch)
            modo = "tarball"
            if resultado_repo.get("erro"):
                print(f"  [codigo] {repo.nome}: {resultado_repo['erro']}")

        arquivos_baixados += ok_repo
        total_bytes += bytes_repo
        subfontes.append(
            {
                "repo": repo.nome,
                "url": repo.url_oficial,
                "linguagem": repo.linguagem,
                "licenca": licenca_real,
                "licenca_confirmada_api": bool(info.get("licenca")),
                "modo": modo,
                "branch": branch,
                "arquivos": ok_repo,
                "bytes": bytes_repo,
            }
        )
        print(
            f"  [codigo] {repo.nome:32s} {repo.linguagem:11s} {ok_repo:3d} arquivos "
            f"{bytes_repo / 1024:8.0f} KB  lic={licenca_real}  ({modo})"
        )

    entrada.tamanho_bytes = total_bytes
    entrada.tamanho_em_disco_bytes = _tamanho_dir(destino_base)
    entrada.amostras = arquivos_baixados
    entrada.subfontes = subfontes
    entrada.status_download = "concluido" if arquivos_baixados else "falhou"
    entrada.status_preparacao = "pendente"
    # Verificação real: os arquivos precisam estar no disco de fato.
    em_disco = sum(1 for p in destino_base.rglob("*") if p.is_file())
    if em_disco != arquivos_baixados:
        entrada.observacoes = (
            f"AVISO: {arquivos_baixados} arquivos contabilizados, mas {em_disco} "
            "encontrados no disco — reexecute o download para reconferir."
        )
        entrada.status_download = "parcial"
    entrada.licenca = (
        "Por repositório (licença declarada pela API do GitHub quando disponível): "
        + ", ".join(
            sorted({s.get("licenca", "?") for s in subfontes if s.get("licenca")})
        )
    )
    manifesto.registrar(entrada)
    return entrada


def preparar_imagens(spec: DatasetSpec, base: Path, manifesto: DatasetManifest, downloader: Downloader) -> EntradaManifesto:
    """Baixa o parquet do Flickr8k e conta as imagens."""
    entrada = manifesto.entradas.get(spec.nome) or _entrada_base(spec)
    entrada.status_download = "em_andamento"
    manifesto.registrar(entrada)

    destino_base = base / "raw"
    total = 0
    detalhes: List[Dict[str, Any]] = []
    for arquivo in spec.arquivos:
        resultado = downloader.baixar_arquivo(arquivo, destino_base)
        detalhes.append(resultado.para_dict())
        if resultado.ok:
            total += resultado.bytes

    amostras = 0
    try:
        import pyarrow.parquet as pq

        for arquivo in spec.arquivos:
            parquet = destino_base / arquivo.destino_relativo
            if parquet.exists():
                amostras += pq.ParquetFile(parquet).metadata.num_rows
    except Exception as exc:
        entrada.observacoes += f" (contagem falhou: {exc})"

    entrada.tamanho_bytes = total
    entrada.tamanho_em_disco_bytes = _tamanho_dir(destino_base)
    entrada.amostras = amostras
    entrada.arquivos = detalhes
    entrada.sha256 = detalhes[0].get("sha256") if detalhes else None
    entrada.status_download = "concluido" if total else "falhou"
    manifesto.registrar(entrada)
    return entrada


def preparar_audio(spec: DatasetSpec, base: Path, manifesto: DatasetManifest, downloader: Downloader) -> EntradaManifesto:
    """Baixa o LibriSpeech e extrai os pares áudio/texto."""
    entrada = manifesto.entradas.get(spec.nome) or _entrada_base(spec)
    entrada.status_download = "em_andamento"
    manifesto.registrar(entrada)

    destino_base = base / "raw"
    total = 0
    detalhes: List[Dict[str, Any]] = []
    for arquivo in spec.arquivos:
        resultado = downloader.baixar_arquivo(arquivo, destino_base)
        detalhes.append(resultado.para_dict())
        if resultado.ok:
            total += resultado.bytes

    extraido = destino_base / "librispeech" / "LibriSpeech"
    if not extraido.exists():
        tar = destino_base / "librispeech" / "dev-clean.tar.gz"
        if tar.exists():
            print("  [audio] extraindo dev-clean.tar.gz ...")
            with tarfile.open(tar, "r:gz") as arquivo_tar:
                arquivo_tar.extractall(destino_base / "librispeech")
            tar.unlink(missing_ok=True)  # o tar.gz não é mais necessário

    falas = len(list(extraido.rglob("*.flac"))) if extraido.exists() else 0
    transcricoes = len(list(extraido.rglob("*.txt"))) if extraido.exists() else 0

    entrada.tamanho_bytes = total
    entrada.tamanho_em_disco_bytes = _tamanho_dir(destino_base)
    entrada.amostras = falas
    entrada.arquivos = detalhes
    entrada.sha256 = detalhes[0].get("sha256") if detalhes else None
    entrada.status_download = "concluido" if total else "falhou"
    entrada.status_preparacao = "concluido" if falas else "pendente"
    entrada.observacoes = f"{falas} arquivos FLAC e {transcricoes} transcrições em disco."
    manifesto.registrar(entrada)
    return entrada


def preparar_video(spec: DatasetSpec, base: Path, manifesto: DatasetManifest, downloader: Downloader) -> EntradaManifesto:
    """Baixa vídeos livres do Wikimedia Commons."""
    entrada = manifesto.entradas.get(spec.nome) or _entrada_base(spec)
    entrada.status_download = "em_andamento"
    manifesto.registrar(entrada)

    destino_base = base / "raw"
    videos = listar_videos_commons(limite=25, max_size_mb=25.0)
    total = 0
    detalhes: List[Dict[str, Any]] = []
    subfontes: List[Dict[str, Any]] = []
    for video in videos:
        if not video.get("url"):
            continue
        nome = video["titulo"].replace("File:", "").replace(" ", "_")
        destino = destino_base / nome
        resultado = downloader.baixar(
            video["url"], destino, tamanho_esperado=video.get("size")
        )
        detalhes.append(resultado.para_dict())
        if resultado.ok:
            total += resultado.bytes
        subfontes.append(
            {
                "titulo": video["titulo"],
                "url": video["url"],
                "licenca": video.get("licenca"),
                "autor": video.get("autor"),
                "mime": video.get("mime"),
                "bytes": resultado.bytes,
            }
        )
        time.sleep(0.2)

    entrada.tamanho_bytes = total
    entrada.tamanho_em_disco_bytes = _tamanho_dir(destino_base)
    # Conta também os vídeos que já estavam no disco (ex.: de uma rodada
    # anterior com um limite de tamanho diferente).
    from datasets.video_loader import descobrir_videos

    videos_em_disco = descobrir_videos(destino_base)
    entrada.amostras = len(videos_em_disco)
    entrada.arquivos = detalhes
    entrada.subfontes = subfontes
    entrada.status_download = "concluido" if entrada.amostras else "falhou"
    entrada.observacoes = (
        f"{entrada.amostras} vídeos em disco nesta rodada; "
        f"{len([d for d in detalhes if d.get('ok')])} baixados agora."
    )
    manifesto.registrar(entrada)
    return entrada


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Download verificado de datasets multimodais")
    parser.add_argument("--datasets", nargs="*", default=[], help="nomes do catálogo")
    parser.add_argument("--modalidades", nargs="*", default=[], help="modalidades inteiras")
    parser.add_argument("--listar", action="store_true", help="lista o catálogo e sai")
    parser.add_argument("--manifesto", default="dataset_manifest.json")
    parser.add_argument("--config", default="configs/resource_manager.yaml")
    parser.add_argument("--raiz", default="datasets")
    args = parser.parse_args()

    if args.listar:
        for spec in listar():
            print(f"[{spec.modalidade:8s}] {spec.nome:30s} {spec.licenca}")
        return

    cfg = carregar_config_recursos(args.config)
    downloader = Downloader(cfg, raiz=RAIZ)
    manifesto = DatasetManifest(RAIZ / args.manifesto)

    selecionados: List[DatasetSpec] = []
    for nome in args.datasets:
        selecionados.append(obter(nome))
    for modalidade in args.modalidades:
        selecionados.extend(listar(modalidade))

    if not selecionados:
        print("nenhum dataset selecionado; use --datasets ou --modalidades")
        return

    preparadores = {
        "texto": preparar_texto,
        "codigo": preparar_codigo,
        "images": preparar_imagens,
        "audio": preparar_audio,
        "video": preparar_video,
    }

    for spec in selecionados:
        base = RAIZ / args.raiz / spec.modalidade
        base.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {spec.nome} ({spec.modalidade}) ===")
        preparador = preparadores.get(spec.modalidade)
        if preparador is None:
            print(f"  sem preparador para a modalidade '{spec.modalidade}'")
            continue
        entrada = preparador(spec, base, manifesto, downloader)
        print(
            f"  -> status={entrada.status_download} "
            f"amostras={entrada.amostras} "
            f"bytes={entrada.tamanho_bytes}"
        )

    print("\n[manifesto] resumo:")
    print(json.dumps(manifesto.resumo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
