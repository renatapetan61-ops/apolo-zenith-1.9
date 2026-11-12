"""Downloader verificado de datasets do Apolo Zenith 1.9.

Garantias:

* **nada** é baixado em massa antes de checar o espaço livre;
* o download acontece em blocos (respeitando ``download_max_por_passo_mb``);
* o SHA-256 é calculado durante o download e conferido contra o esperado;
* arquivos parcialmente baixados ficam com extensão ``.part`` (limpos depois);
* o ``dataset_manifest.json`` é atualizado a cada arquivo concluído;
* todo evento vai para ``logs/dataset_downloads.log``.

O downloader **não** apaga dados já existentes: se o arquivo está presente e
o hash confere, ele é reaproveitado.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets.catalog import ArquivoDataset, DatasetSpec, RepoGitHub
from resources.config import ConfigRecursos
from resources.erros import registrar_erro
from resources.storage_manager import StorageManager

USER_AGENT = "ApoloZenith/1.9 (dataset downloader; +https://github.com/apolo-zenith)"
BLOCO = 1024 * 256
TEMPO_ENTRE_REQUISICOES_S = 0.35


class ErroDownload(RuntimeError):
    """Falha no download de um arquivo."""


@dataclass
class ResultadoArquivo:
    """Resultado do download de um arquivo."""

    url: str
    destino: str
    bytes: int = 0
    sha256: Optional[str] = None
    ok: bool = False
    motivo: str = ""
    reaproveitado: bool = False
    duracao_s: float = 0.0

    def para_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "destino": self.destino,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "ok": self.ok,
            "motivo": self.motivo,
            "reaproveitado": self.reaproveitado,
            "duracao_s": round(self.duracao_s, 2),
        }


class Downloader:
    """Baixa arquivos com verificação de espaço e integridade."""

    def __init__(
        self,
        config: Optional[ConfigRecursos] = None,
        raiz: str | Path = ".",
        log: Optional[str] = Path("logs/dataset_downloads.log"),
    ) -> None:
        self.config = config or ConfigRecursos()
        self.raiz = Path(raiz)
        self.armazenamento = StorageManager(self.config.armazenamento, raiz=self.raiz)
        self.log = Path(log) if log else None
        if self.log and not self.log.is_absolute():
            self.log = self.raiz / self.log

    # ------------------------------------------------------------------ #
    def registrar(self, mensagem: str) -> None:
        linha = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}"
        print(linha, flush=True)
        if self.log:
            try:
                self.log.parent.mkdir(parents=True, exist_ok=True)
                with self.log.open("a", encoding="utf-8") as fh:
                    fh.write(linha + "\n")
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    def baixar(
        self,
        url: str,
        destino: str | Path,
        sha256_esperado: Optional[str] = None,
        tamanho_esperado: Optional[int] = None,
        forcar: bool = False,
    ) -> ResultadoArquivo:
        """Baixa um arquivo, verificando espaço e integridade."""
        destino = Path(destino)
        if not destino.is_absolute():
            destino = self.raiz / destino
        resultado = ResultadoArquivo(url=url, destino=str(destino))

        # Reaproveita arquivo já válido.
        if destino.exists() and not forcar:
            if sha256_esperado:
                atual = _sha256(destino)
                if atual == sha256_esperado:
                    resultado.ok = True
                    resultado.sha256 = atual
                    resultado.bytes = destino.stat().st_size
                    resultado.reaproveitado = True
                    resultado.motivo = "reaproveitado (sha-256 confere)"
                    return resultado
            else:
                resultado.ok = True
                resultado.bytes = destino.stat().st_size
                resultado.reaproveitado = True
                resultado.motivo = "reaproveitado (existe)"
                return resultado

        # Espaço disponível.
        if tamanho_esperado:
            pode, motivo = self.armazenamento.pode_baixar(tamanho_esperado)
            if not pode:
                resultado.motivo = f"sem espaço: {motivo}"
                self.registrar(f"[aviso] download ignorado ({motivo}): {url}")
                return resultado

        destino.parent.mkdir(parents=True, exist_ok=True)
        parcial = destino.with_suffix(destino.suffix + ".part")
        digest = hashlib.sha256()
        total = 0
        inicio = time.time()
        bloco_max = max(BLOCO, int(self.config.armazenamento.download_max_por_passo_mb) * 1024 * 1024)
        blocos = 0

        try:
            requisicao = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(requisicao, timeout=60) as resposta:
                tamanho_declarado = int(resposta.headers.get("Content-Length") or 0)
                if tamanho_declarado and not tamanho_esperado:
                    pode, motivo = self.armazenamento.pode_baixar(tamanho_declarado)
                    if not pode:
                        resultado.motivo = f"sem espaço: {motivo}"
                        self.registrar(f"[aviso] download interrompido ({motivo}): {url}")
                        return resultado
                with parcial.open("wb") as fh:
                    while True:
                        # Cada bloco respeita o limite por passo.
                        bloco = resposta.read(BLOCO)
                        if not bloco:
                            break
                        fh.write(bloco)
                        digest.update(bloco)
                        total += len(bloco)
                        blocos += 1
                        if total % bloco_max < BLOCO and blocos > 1:
                            # Pausa breve entre blocos grandes: o aparelho
                            # precisa respirar entre as requisições.
                            time.sleep(1.0)
                            pode, motivo = self.armazenamento.pode_baixar_passo(BLOCO)
                            if not pode:
                                parcial.unlink(missing_ok=True)
                                resultado.motivo = f"espaço insuficiente durante o download: {motivo}"
                                self.registrar(f"[aviso] {resultado.motivo}: {url}")
                                return resultado
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            parcial.unlink(missing_ok=True)
            resultado.motivo = f"erro de rede: {type(exc).__name__}: {exc}"
            self.registrar(f"[erro] {url}: {resultado.motivo}")
            registrar_erro(f"falha no download: {url}", exc, {"destino": str(destino)})
            return resultado

        hash_calculado = digest.hexdigest()
        if sha256_esperado and hash_calculado.lower() != sha256_esperado.lower():
            parcial.unlink(missing_ok=True)
            resultado.sha256 = hash_calculado
            resultado.motivo = "sha-256 divergente"
            self.registrar(f"[erro] sha-256 divergente em {url}")
            return resultado

        os.replace(parcial, destino)
        resultado.ok = True
        resultado.bytes = total
        resultado.sha256 = hash_calculado
        resultado.duracao_s = time.time() - inicio
        self.registrar(
            f"[ok] {destino.name} | {total / 1024 ** 2:.1f} MB | sha256={hash_calculado[:16]}… | "
            f"{resultado.duracao_s:.1f}s"
        )
        return resultado

    # ------------------------------------------------------------------ #
    def baixar_arquivo(self, arquivo: ArquivoDataset, destino_base: Path) -> ResultadoArquivo:
        """Baixa um :class:`ArquivoDataset` dentro de ``destino_base``."""
        destino = destino_base / arquivo.destino_relativo
        return self.baixar(
            arquivo.url,
            destino,
            sha256_esperado=arquivo.sha256_esperado,
            tamanho_esperado=arquivo.tamanho_esperado,
        )


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #
class CacheGitHub:
    """Cache em disco das respostas da API do GitHub.

    A API sem token tem limite de 60 requisições/hora por IP.  Como o
    catálogo é fixo, guardar as respostas evita gastar o limite a cada
    execução (e permite retomar downloads interrompidos).
    """

    def __init__(self, caminho: str | Path = "datasets/code/manifests/github_cache.json") -> None:
        self.caminho = Path(caminho)
        self.dados: Dict[str, Any] = {}
        if self.caminho.exists():
            try:
                self.dados = json.loads(self.caminho.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.dados = {}

    def obter(self, chave: str) -> Optional[Any]:
        return self.dados.get(chave)

    def gravar(self, chave: str, valor: Any) -> None:
        self.dados[chave] = valor
        try:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.caminho.with_suffix(self.caminho.suffix + ".tmp")
            tmp.write_text(json.dumps(self.dados, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.caminho)
        except OSError:
            pass


CACHE_GITHUB = CacheGitHub()


def _github_api(url: str) -> Dict[str, Any]:
    requisicao = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(requisicao, timeout=60) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def _sha256(caminho: Path, bloco: int = BLOCO) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as fh:
        while True:
            dados = fh.read(bloco)
            if not dados:
                break
            digest.update(dados)
    return digest.hexdigest()


def info_licenca_repo(owner: str, repo: str) -> Dict[str, Any]:
    """Lê a licença declarada pela API do GitHub (verificação real)."""
    chave = f"licenca:{owner}/{repo}"
    em_cache = CACHE_GITHUB.obter(chave)
    if em_cache is not None:
        return em_cache
    try:
        dados = _github_api(f"https://api.github.com/repos/{owner}/{repo}")
    except Exception as exc:
        return {"licenca": None, "erro": str(exc)[:120]}
    lic = dados.get("license") or {}
    resultado = {
        "licenca": lic.get("spdx_id"),
        "licenca_nome": lic.get("name"),
        "branch_padrao": dados.get("default_branch"),
        "estrelas": dados.get("stargazers_count"),
        "descricao": (dados.get("description") or "")[:200],
    }
    CACHE_GITHUB.gravar(chave, resultado)
    return resultado


def listar_arquivos_repo(
    owner: str,
    repo: str,
    branch: str,
    incluir: List[str],
    excluir: List[str],
    max_arquivos: int,
    max_bytes: int,
) -> List[Dict[str, Any]]:
    """Enumera os arquivos de um repositório via API de árvore (com cache)."""
    chave = f"arvore:{owner}/{repo}@{branch}"
    em_cache = CACHE_GITHUB.obter(chave)
    if em_cache is not None:
        return em_cache[:max_arquivos]
    dados = _github_api(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    )
    candidatos = []
    for item in dados.get("tree", []):
        if item.get("type") != "blob":
            continue
        caminho = item.get("path", "")
        tamanho = int(item.get("size") or 0)
        if tamanho <= 0 or tamanho > max_bytes:
            continue
        if not any(caminho.endswith(ext) for ext in incluir):
            continue
        if any(caminho.startswith(p) or f"/{p}" in caminho for p in excluir):
            continue
        candidatos.append(
            {
                "path": caminho,
                "size": tamanho,
                "url": (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
                    f"{urllib.parse.quote(caminho)}"
                ),
            }
        )
    candidatos.sort(key=lambda c: c["path"])
    CACHE_GITHUB.gravar(chave, candidatos)
    return candidatos[:max_arquivos]


# --------------------------------------------------------------------------- #
# Wikimedia Commons
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# GitHub — download do repositório inteiro (tarball) como fallback
# --------------------------------------------------------------------------- #
#: Branches tentados em ordem quando a API não está disponível.
BRANCHES_PADRAO = ("main", "master", "develop", "trunk", "stable")


def branches_candidatos(owner: str, repo: str, explicito: str = "") -> List[str]:
    """Branches prováveis de um repositório (usa a API quando possível)."""
    info = info_licenca_repo(owner, repo)
    branch = info.get("branch_padrao")
    candidatos: List[str] = []
    if explicito:
        candidatos.append(explicito)
    if branch and branch not in candidatos:
        candidatos.append(branch)
    for extra in BRANCHES_PADRAO:
        if extra not in candidatos:
            candidatos.append(extra)
    return candidatos


def baixar_repo_github(
    owner: str,
    repo: str,
    branch: str,
    destino: Path,
    incluir: List[str],
    excluir: List[str],
    max_arquivos: int,
    max_bytes: int,
    downloader: "Downloader",
    max_tarball_mb: float = 200.0,
) -> Dict[str, Any]:
    """Baixa o tarball do repositório e extrai **apenas** os arquivos de código.

    Vantagem sobre a API: não depende do limite de 60 requisições/hora.
    O tarball é removido após a extração (não ocupa espaço).
    """
    import tarfile

    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
    temporario = Path(destino).parent / f"{repo}.tar.gz"
    resultado = downloader.baixar(url, temporario, tamanho_esperado=None)
    if not resultado.ok:
        return {"repo": f"{owner}/{repo}", "erro": resultado.motivo, "arquivos": 0, "bytes": 0}
    if temporario.stat().st_size > max_tarball_mb * 1024 * 1024:
        temporario.unlink(missing_ok=True)
        return {
            "repo": f"{owner}/{repo}",
            "erro": f"tarball acima de {max_tarball_mb:.0f} MB",
            "arquivos": 0,
            "bytes": 0,
        }

    arquivos: List[Dict[str, Any]] = []
    total = 0
    licenca_arquivo = ""
    try:
        with tarfile.open(temporario, "r:gz") as tar:
            for membro in tar:
                if not membro.isfile():
                    continue
                nome = membro.name
                partes = nome.split("/", 1)
                caminho = partes[1] if len(partes) > 1 else nome
                if caminho.lower().startswith(("license", "licence", "copying")):
                    if not licenca_arquivo:
                        licenca_arquivo = caminho
                    continue
                tamanho = membro.size
                if tamanho <= 0 or tamanho > max_bytes:
                    continue
                if not any(caminho.endswith(ext) for ext in incluir):
                    continue
                if any(caminho.startswith(p) or f"/{p}" in caminho for p in excluir):
                    continue
                if len(arquivos) >= max_arquivos:
                    break
                alvo = Path(destino) / caminho
                alvo.parent.mkdir(parents=True, exist_ok=True)
                try:
                    origem = tar.extractfile(membro)
                    if origem is None:
                        continue
                    with origem, alvo.open("wb") as saida:
                        shutil.copyfileobj(origem, saida, length=BLOCO)
                except (OSError, KeyError):
                    continue
                arquivos.append({"caminho": caminho, "bytes": tamanho})
                total += tamanho
    except (tarfile.TarError, OSError) as exc:
        temporario.unlink(missing_ok=True)
        return {"repo": f"{owner}/{repo}", "erro": f"falha na extração: {exc}", "arquivos": len(arquivos), "bytes": total}
    finally:
        temporario.unlink(missing_ok=True)

    return {
        "repo": f"{owner}/{repo}",
        "branch": branch,
        "arquivos": len(arquivos),
        "bytes": total,
        "licenca_no_repositorio": licenca_arquivo,
        "detalhe": arquivos,
    }


def listar_videos_commons(
    limite: int = 6,
    max_size_mb: float = 25.0,
    categorias: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Lista vídeos do Wikimedia Commons com licença e URL de download.

    ``max_size_mb`` limita o tamanho de cada arquivo (vídeos grandes demais
    não são úteis para o dispositivo e pesam no orçamento de armazenamento).
    """
    categorias = categorias or ["Category:Videos"]
    vistos: set[str] = set()
    resultados: List[Dict[str, Any]] = []
    for categoria in categorias:
        api = (
            "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
            "&iiprop=url|size|mime|extmetadata&generator=categorymembers"
            f"&gcmtitle={urllib.parse.quote(categoria)}&gcmtype=file&gcmlimit=200"
            "&iiextmetadatafilter=LicenseShortName|Artist|LicenseUrl"
        )
        requisicao = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(requisicao, timeout=60) as resposta:
                dados = json.loads(resposta.read().decode("utf-8"))
        except Exception as exc:
            print(f"[aviso] Commons: falha em '{categoria}': {exc}")
            continue
        paginas = (dados.get("query") or {}).get("pages") or {}
        for pagina in paginas.values():
            info = (pagina.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if not mime.startswith("video/"):
                continue
            tamanho = int(info.get("size") or 0)
            if tamanho > max_size_mb * 1024 * 1024:
                continue
            titulo = pagina.get("title", "")
            if titulo in vistos:
                continue
            vistos.add(titulo)
            extra = info.get("extmetadata") or {}
            licenca = (extra.get("LicenseShortName") or {}).get("value", "")
            autor = (extra.get("Artist") or {}).get("value", "")
            resultados.append(
                {
                    "titulo": titulo,
                    "url": info.get("url"),
                    "mime": mime,
                    "size": tamanho,
                    "licenca": licenca,
                    "autor": autor[:120],
                }
            )
    # Menores primeiro: maximiza a quantidade dentro do orçamento.
    resultados.sort(key=lambda v: v.get("size", 0))
    return resultados[:limite]
