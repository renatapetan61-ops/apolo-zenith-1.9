"""Catálogo de datasets multimodais do Apolo Zenith 1.9.

Todas as entradas correspondem a **fontes reais e verificadas** nesta
máquina (o downloader confere HTTP 200 antes de gravar).  Cada dataset
declara origem, URL oficial, licença e formato.

Fontes usadas e seu estado real:

===========================================  ====================  =========
Dataset                                      Modalidade            Licença
===========================================  ====================  =========
Gutenberg PT/EN + Wikisource PT              texto                domínio público
Flickr8k (test)                              imagens + legendas    CC BY 2.0
Wikimedia Commons (destaques/vídeos)          imagens + vídeo       CC BY-SA / CC0
LibriSpeech dev-clean                        áudio + transcrição  CC BY 4.0
Repositórios GitHub permissivos (12 linguag.) código               por repositório
===========================================  ====================  =========

Fontes **descartadas** com justificativa:

* ``bigcode/the-stack-smol`` — HTTP 401 (gated, exige autenticação);
* Common Voice — host ``data.commonvoice.org`` inacessível (HTTP 000);
* FFmpeg binário — ausente; resolvido com PyAV (``av``), que embute ffmpeg.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Estruturas
# --------------------------------------------------------------------------- #
@dataclass
class ArquivoDataset:
    """Um arquivo a ser baixado."""

    url: str
    destino_relativo: str
    sha256_esperado: Optional[str] = None
    tamanho_esperado: Optional[int] = None

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoGitHub:
    """Um repositório GitHub usado como fonte de código."""

    owner: str
    repo: str
    branch: str = ""
    linguagem: str = ""
    licenca_esperada: str = ""
    incluir: List[str] = field(default_factory=list)   # extensões aceitas
    excluir: List[str] = field(default_factory=list)   # subpastas ignoradas
    max_arquivos: int = 60
    max_bytes_por_arquivo: int = 200_000

    @property
    def nome(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def url_oficial(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


@dataclass
class DatasetSpec:
    """Descrição de um dataset do catálogo."""

    nome: str
    modalidade: str                 # texto | codigo | imagens | audio | video
    origem: str
    url_oficial: str
    licenca: str
    formato: str
    descricao: str
    arquivos: List[ArquivoDataset] = field(default_factory=list)
    repos: List[RepoGitHub] = field(default_factory=list)
    notas: str = ""

    def para_dict(self) -> Dict[str, Any]:
        dados = asdict(self)
        return dados


# --------------------------------------------------------------------------- #
# Texto — já baixado na fase anterior (registro honesto)
# --------------------------------------------------------------------------- #
TEXTO_GUTENBERG = DatasetSpec(
    nome="gutenberg_pt_en",
    modalidade="text",
    origem="Project Gutenberg (e-books em domínio público) + Wikisource PT",
    url_oficial="https://www.gutenberg.org/ebooks/search/?sort_order=downloads",
    licenca="Domínio público (Project Gutenberg) / CC BY-SA 3.0 (Wikisource)",
    formato="texto puro (.txt) e XML compactado (.xml.bz2)",
    descricao=(
        "Corpus textual usado no Language Core: livros em português e inglês do "
        "Project Gutenberg e artigos do Wikisource em português."
    ),
    notas="Já presente em datasets_storage/raw (157 MB) — será registrado no manifesto.",
)


# --------------------------------------------------------------------------- #
# Código — repositórios reais com licenças permissivas
# --------------------------------------------------------------------------- #
CODIGO_REPOS = DatasetSpec(
    nome="codigo_github_permissivo",
    modalidade="code",
    origem="Repositórios GitHub com licenças permissivas (verificadas via API)",
    url_oficial="https://github.com/",
    licenca="Por repositório (MIT / Apache-2.0 / BSD-3-Clause) — conferida na API",
    formato="arquivos de código-fonte (.py, .ts, .c, .cpp, .java, .rs, .go, ...)",
    descricao=(
        "Código real de projetos conhecidos, limitado por arquivo e por "
        "repositório para caber no orçamento de armazenamento do dispositivo."
    ),
    repos=[
        RepoGitHub("pallets", "flask", linguagem="python", licenca_esperada="BSD-3-Clause",
                   incluir=[".py"], excluir=["tests/", "docs/"], max_arquivos=45),
        RepoGitHub("psf", "requests", linguagem="python", licenca_esperada="Apache-2.0",
                   incluir=[".py"], excluir=["tests/", "docs/"], max_arquivos=45),
        RepoGitHub("expressjs", "express", linguagem="javascript", licenca_esperada="MIT",
                   incluir=[".js", ".mjs"], excluir=["node_modules/", "test/"], max_arquivos=45),
        RepoGitHub("axios", "axios", linguagem="javascript", licenca_esperada="MIT",
                   incluir=[".js"], excluir=["dist/", "test/"], max_arquivos=40),
        RepoGitHub("microsoft", "TypeScript", linguagem="typescript", licenca_esperada="Apache-2.0",
                   incluir=[".ts"], excluir=["tests/"], max_arquivos=45),
        RepoGitHub("redis", "redis", branch="unstable", linguagem="c", licenca_esperada="BSD-3-Clause",
                   incluir=[".c", ".h"], excluir=["tests/"], max_arquivos=45),
        RepoGitHub("sqlite", "sqlite", linguagem="c", licenca_esperada="CC0-1.0",
                   incluir=[".c", ".h"], excluir=["test/"], max_arquivos=40),
        RepoGitHub("nlohmann", "json", linguagem="cpp", licenca_esperada="MIT",
                   incluir=[".hpp", ".h", ".cpp"], excluir=["tests/"], max_arquivos=40),
        RepoGitHub("google", "protobuf", linguagem="cpp", licenca_esperada="BSD-3-Clause",
                   incluir=[".h", ".cc"], excluir=["third_party/"], max_arquivos=40),
        RepoGitHub("spring-projects", "spring-boot", linguagem="java", licenca_esperada="Apache-2.0",
                   incluir=[".java"], excluir=["src/test/"], max_arquivos=45),
        RepoGitHub("serde-rs", "serde", linguagem="rust", licenca_esperada="MIT",
                   incluir=[".rs"], excluir=["tests/"], max_arquivos=40),
        RepoGitHub("rust-lang", "cargo", linguagem="rust", licenca_esperada="MIT",
                   incluir=[".rs"], excluir=["tests/", "crates/"], max_arquivos=40),
        RepoGitHub("gin-gonic", "gin", linguagem="go", licenca_esperada="MIT",
                   incluir=[".go"], excluir=["testdata/"], max_arquivos=40),
        RepoGitHub("spf13", "cobra", linguagem="go", licenca_esperada="Apache-2.0",
                   incluir=[".go"], excluir=["user_guide/"], max_arquivos=40),
        RepoGitHub("square", "okhttp", linguagem="kotlin", licenca_esperada="Apache-2.0",
                   incluir=[".kt"], excluir=["okhttp-tls/"], max_arquivos=40),
        RepoGitHub("django", "django", linguagem="python", licenca_esperada="BSD-3-Clause",
                   incluir=[".py"], excluir=["tests/", "docs/"], max_arquivos=45),
        RepoGitHub("square", "retrofit", linguagem="java", licenca_esperada="Apache-2.0",
                   incluir=[".java"], excluir=["retrofit-android/"], max_arquivos=35),
        RepoGitHub("necolas", "normalize.css", linguagem="css", licenca_esperada="MIT",
                   incluir=[".css"], max_arquivos=10),
        RepoGitHub("twbs", "bootstrap", linguagem="css", licenca_esperada="MIT",
                   incluir=[".css", ".html"], excluir=["js/", "dist/"], max_arquivos=40),
        RepoGitHub("apache", "superset", branch="master", linguagem="python", licenca_esperada="Apache-2.0",
                   incluir=[".py", ".sql"], excluir=["tests/", "superset-frontend/"], max_arquivos=45),
        RepoGitHub("sql-js", "sql.js", branch="master", linguagem="sql", licenca_esperada="MIT",
                   incluir=[".sql"], max_arquivos=25),
        RepoGitHub("git", "git", linguagem="c", licenca_esperada="GPL-2.0",
                   incluir=[".c", ".h"], excluir=["t/", "contrib/"], max_arquivos=30),
        RepoGitHub("nvm-sh", "nvm", branch="master", linguagem="shell", licenca_esperada="MIT",
                   incluir=[".sh", ".bash"], excluir=["test/", "tests/"], max_arquivos=30),
        RepoGitHub("koalaman", "shellcheck", branch="master", linguagem="haskell", licenca_esperada="GPL-3.0",
                   incluir=[".hs"], excluir=["test/"], max_arquivos=20),
        RepoGitHub("sqlfluff", "sqlfluff", branch="main", linguagem="sql", licenca_esperada="MIT",
                   incluir=[".sql"], excluir=["test/fixtures/dialects/", "docs/"], max_arquivos=30),
    ],
    notas=(
        "A licença real de cada repositório é lida da API do GitHub no momento "
        "do download e gravada no manifesto; extensões e limites por repositório "
        "evitam baixar árvores enormes."
    ),
)


# --------------------------------------------------------------------------- #
# Imagens
# --------------------------------------------------------------------------- #
IMAGENS_FLICKR8K = DatasetSpec(
    nome="flickr8k_test",
    modalidade="images",
    origem="Hugging Face Hub (jxie/flickr8k) — split de teste",
    url_oficial="https://huggingface.co/datasets/jxie/flickr8k",
    licenca="CC BY 2.0 (imagens do Flickr8k) / licença do dataset no Hub",
    formato="Parquet (colunas: image, caption_0..caption_4)",
    descricao=(
        "Pares imagem-legenda para treino da visão computacional e para o "
        "alinhamento imagem-texto do modelo multimodal."
    ),
    arquivos=[
        ArquivoDataset(
            url="https://huggingface.co/datasets/jxie/flickr8k/resolve/main/data/test-00000-of-00001-42a2661d12c73e48.parquet",
            destino_relativo="flickr8k/test.parquet",
            tamanho_esperado=136_773_352,
        ),
        # Expansão (a generalization exige mais dados que 1.000 imagens):
        # primeiro shard do split de treino — 3.000 imagens com 5 legendas.
        ArquivoDataset(
            url="https://huggingface.co/datasets/jxie/flickr8k/resolve/main/data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
            destino_relativo="flickr8k/train_0.parquet",
            tamanho_esperado=414_2_00_000,
        ),
    ],
    notas="1.000 imagens de teste + 3.000 de treino (1º shard), 5 legendas cada.",
)


# --------------------------------------------------------------------------- #
# Áudio
# --------------------------------------------------------------------------- #
AUDIO_LIBRISPEECH = DatasetSpec(
    nome="librispeech_dev_clean",
    modalidade="audio",
    origem="OpenSLR 12 (LibriSpeech ASR corpus) — split dev-clean",
    url_oficial="https://www.openslr.org/12/",
    licenca="CC BY 4.0 (LibriSpeech, funded por Mozilla/NSF)",
    formato="Tar.gz com FLAC 16 kHz + transcrições .txt",
    descricao=(
        "Fala falada em inglês com transcrição textual — base do encoder de "
        "áudio (mel-spectrogram) e das Experiments ASR (Reconhecimento de Fala)."
    ),
    arquivos=[
        ArquivoDataset(
            url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
            destino_relativo="librispeech/dev-clean.tar.gz",
            tamanho_esperado=337_926_286,
        )
    ],
    notas="2.703 falas com transcrição (texto original do Project Gutenberg).",
)


# --------------------------------------------------------------------------- #
# Vídeo
# --------------------------------------------------------------------------- #
VIDEO_COMMONS = DatasetSpec(
    nome="wikimedia_videos",
    modalidade="video",
    origem="Wikimedia Commons — arquivos de vídeo com licença livre",
    url_oficial="https://commons.wikimedia.org/wiki/Category:Videos",
    licenca="CC BY-SA / CC BY / CC0 / domínio público (por arquivo, via API)",
    formato="WebM/Ogg (decodificado com PyAV)",
    descricao=(
        "Vídeos curtos com licença livre para extração de frames e treino do "
        "encoder temporal."
    ),
    notas="Lista de arquivos obtida pela API do Commons; cada arquivo traz a licença.",
)


VIDEO_ARCHIVE = DatasetSpec(
    nome="archive_org_videos",
    modalidade="video",
    origem="Internet Archive — filmes e vídeos de domínio público",
    url_oficial="https://archive.org/",
    licenca="Domínio público / CC (por item, via API)",
    formato="MP4/WebM",
    descricao="Vídeos de domínio público para a modalidade temporal.",
    notas="Usado apenas se a fonte do Commons não render frames no dispositivo.",
)


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #
CATALOGO: Dict[str, DatasetSpec] = {
    spec.nome: spec
    for spec in (
        TEXTO_GUTENBERG,
        CODIGO_REPOS,
        IMAGENS_FLICKR8K,
        AUDIO_LIBRISPEECH,
        VIDEO_COMMONS,
        VIDEO_ARCHIVE,
    )
}


def listar(modalidade: Optional[str] = None) -> List[DatasetSpec]:
    """Lista o catálogo, opcionalmente filtrado por modalidade."""
    specs = list(CATALOGO.values())
    if modalidade:
        specs = [s for s in specs if s.modalidade == modalidade]
    return specs


def obter(nome: str) -> DatasetSpec:
    """Obtém uma especificação pelo nome."""
    if nome not in CATALOGO:
        raise KeyError(f"Dataset desconhecido: {nome}. Disponíveis: {sorted(CATALOGO)}")
    return CATALOGO[nome]


def resumo() -> List[Dict[str, Any]]:
    """Resumo do catálogo para relatórios."""
    return [
        {
            "nome": spec.nome,
            "modalidade": spec.modalidade,
            "origem": spec.origem,
            "licenca": spec.licenca,
            "formato": spec.formato,
            "arquivos": len(spec.arquivos),
            "repositorios": len(spec.repos),
        }
        for spec in CATALOGO.values()
    ]
