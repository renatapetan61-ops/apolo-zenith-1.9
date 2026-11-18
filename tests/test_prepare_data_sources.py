from scripts import prepare_data

def test_processar_preserva_fonte_wikipedia_sem_limpeza_gutenberg(tmp_path, monkeypatch):
    arquivo=tmp_path/"artigo.txt"
    texto=("A biblioteca preserva documentos históricos e amplia pesquisas científicas. "
        "Pesquisadores comparam fontes, registram autores e verificam datas relevantes. "
        "Estudantes consultam mapas, relatórios, livros e registros culturais diversos. "
        "A tecnologia organiza arquivos, protege dados e facilita novas descobertas. "
        "Cada comunidade compartilha conhecimentos sobre natureza, saúde e educação.")
    arquivo.write_text(texto,encoding="utf-8")
    fontes=[]
    original=prepare_data.normalizar_texto
    def observar(conteudo,fonte="gutenberg"):
        fontes.append(fonte)
        return original(conteudo,fonte=fonte)
    monkeypatch.setattr(prepare_data,"normalizar_texto",observar)
    documentos=[{"registro":{"fonte":"wikipedia_pt"},"caminho":arquivo}]
    processados,stats=prepare_data.processar(documentos,tmp_path)
    assert fontes==["wikipedia"]
    assert stats["aprovados"]==1
    assert processados and processados[0].startswith("A biblioteca preserva")
