"""Avaliação holdout bilíngue e diagnóstico de qualidade do Language Core."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from collections import Counter
from pathlib import Path
import torch
from core.language_model import ApoloLanguageModel
from tokenizer.tokenizer import BPETokenizer

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def load_corpus(path, language):
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows)<6: raise ValueError(f"Corpus {language} precisa de pelo menos 6 documentos.")
    texts=[str(x["text"]).strip() for x in rows]
    if any(len(x)<180 for x in texts): raise ValueError(f"Documento curto demais em {language}.")
    return rows,texts

@torch.inference_mode()
def score(model, tokenizer, docs):
    total_nll=0.0; total_tokens=0; per_doc=[]
    block=model.config.block_size
    for doc in docs:
        ids=tokenizer.encode(doc,add_bos=True,add_eos=True)
        doc_nll=0.0; doc_n=0
        for start in range(0,len(ids)-1,block):
            chunk=ids[start:start+block+1]
            if len(chunk)<2: continue
            x=torch.tensor(chunk[:-1],dtype=torch.long).unsqueeze(0)
            y=torch.tensor(chunk[1:],dtype=torch.long).unsqueeze(0)
            _,loss=model(x,y)
            n=y.numel(); value=float(loss.item())
            if not math.isfinite(value): raise RuntimeError("Loss não finita.")
            doc_nll+=value*n; doc_n+=n
        total_nll+=doc_nll; total_tokens+=doc_n
        if doc_n: per_doc.append({"tokens":doc_n,"loss":round(doc_nll/doc_n,6)})
    mean=total_nll/max(1,total_tokens)
    return {"loss":round(mean,6),"perplexity":round(math.exp(min(mean,30)),6),
            "tokens_scored":total_tokens,"documents":len(per_doc),
            "document_loss_median":round(statistics.median(x["loss"] for x in per_doc),6)}

def repetition_metrics(ids):
    ids=list(ids)
    if not ids: return {"distinct_1":0.0,"distinct_2":0.0,"repeated_token_ratio":1.0,"max_token_run":0,"repeated_3gram_ratio":1.0}
    uni=len(set(ids))/len(ids)
    bi=[tuple(ids[i:i+2]) for i in range(max(0,len(ids)-1))]
    tri=[tuple(ids[i:i+3]) for i in range(max(0,len(ids)-2))]
    maxrun=run=0; prev=None
    for token in ids:
        run=run+1 if token==prev else 1; maxrun=max(maxrun,run); prev=token
    return {"distinct_1":round(uni,4),"distinct_2":round(len(set(bi))/max(1,len(bi)),4),
            "repeated_token_ratio":round(1-uni,4),"max_token_run":maxrun,
            "repeated_3gram_ratio":round(1-len(set(tri))/max(1,len(tri)),4)}

@torch.inference_mode()
def generate(model, tokenizer, prompt, count):
    ids=tokenizer.encode(prompt,add_bos=True)
    if len(ids)>=model.config.block_size: ids=ids[-model.config.block_size+1:]
    inp=torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    out=model.generate(inp,max_new_tokens=count,temperature=0.8,top_k=40,top_p=0.9,
                       repetition_penalty=1.1,use_cache=True)
    new=out[0].tolist()
    return {"text":tokenizer.decode(new),"token_ids":new,**repetition_metrics(new)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-text",required=True); p.add_argument("--checkpoint-code",required=True)
    p.add_argument("--tokenizer",default="tokenizers/tokenizer")
    p.add_argument("--pt-corpus",default="evaluation/corpus/independent_pt_br.jsonl")
    p.add_argument("--en-corpus",default="evaluation/corpus/independent_en.jsonl")
    p.add_argument("--training-summary",default=None); p.add_argument("--data-summary",default=None)
    p.add_argument("--output",default="reports/language-quality-diagnostic.json")
    a=p.parse_args()
    tok=BPETokenizer.load(a.tokenizer)
    text=ApoloLanguageModel.load(a.checkpoint_text,map_location="cpu").eval()
    code=ApoloLanguageModel.load(a.checkpoint_code,map_location="cpu").eval()
    if text.config.vocab_size!=tok.vocab_size or code.config.vocab_size!=tok.vocab_size:
        raise RuntimeError("Tokenizer e checkpoints incompatíveis.")
    corpora={}
    for lang,path in (("pt-BR",a.pt_corpus),("en",a.en_corpus)):
        rows,docs=load_corpus(path,lang)
        corpora[lang]={"path":path,"sha256":sha(path),"documents":len(rows),
          "characters":sum(map(len,docs)),"tokenizer_tokens":sum(len(tok.encode(x)) for x in docs),
          "evaluation":"independent authored holdout; not drawn from training corpus"}
    comparison={}
    for lang,path in (("pt-BR",a.pt_corpus),("en",a.en_corpus)):
        _,docs=load_corpus(path,lang)
        comparison[lang]={"code_checkpoint":score(code,tok,docs),"text_checkpoint":score(text,tok,docs)}
        base=comparison[lang]["code_checkpoint"]["loss"]; after=comparison[lang]["text_checkpoint"]["loss"]
        comparison[lang]["text_loss_change_pct"]=round((after-base)/max(base,1e-12)*100,3)
    prompts={"pt-BR":["A ciência de dados ajuda equipes a","Quando uma comunidade planeja o futuro","Uma explicação clara deve apresentar"],
      "en":["Data science helps teams to","When a community plans for the future","A clear explanation should present"]}
    generations={}
    for lang,items in prompts.items():
        generations[lang]=[]
        for prompt in items:
            for n in (16,32,64):
                result=generate(text,tok,prompt,n); result.pop("token_ids",None)
                generations[lang].append({"prompt":prompt,"requested_tokens":n,**result})
    samples=[x for lang,path in (("pt-BR",a.pt_corpus),("en",a.en_corpus)) for x in load_corpus(path,lang)[1]]
    all_ids=[i for s in samples for i in tok.encode(s)]
    unk=tok.unk_id
    tokenizer_diag={"vocab_size":tok.vocab_size,"unk_id":unk,
      "unknown_token_rate":round(sum(i==unk for i in all_ids)/max(1,len(all_ids)),6) if unk is not None else 0.0,
      "mean_characters_per_token":round(sum(map(len,samples))/max(1,len(all_ids)),4),
      "tokens_in_holdout":len(all_ids),"sha256":sha(Path(a.tokenizer)/"tokenizer.json")}
    metadata={}
    for key,path in (("training_summary",a.training_summary),("data_summary",a.data_summary)):
        if path and Path(path).is_file():
            try: metadata[key]=json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception as exc: metadata[key]={"read_error":str(exc)}
    gates={}
    for lang in ("pt-BR","en"):
        c=comparison[lang]; t=c["text_checkpoint"]; gens=generations[lang]
        gates[lang]={"at_least_500_eval_tokens":t["tokens_scored"]>=500,
          "finite_metrics":math.isfinite(t["loss"]) and math.isfinite(t["perplexity"]),
          "not_more_than_5pct_loss_regression_vs_code":c["text_loss_change_pct"]<=5,
          "all_generation_runs_nonempty":all(bool(g["text"].strip()) for g in gens),
          "no_severe_token_repetition":all(g["max_token_run"]<8 and g["repeated_3gram_ratio"]<0.75 for g in gens)}
    gates["tokenizer_no_unknowns"]=tokenizer_diag["unknown_token_rate"]==0
    gates["text_improves_at_least_one_language"]=any(comparison[l]["text_loss_change_pct"]< -2 for l in ("pt-BR","en"))
    approved=all(all(v.values()) if isinstance(v,dict) else v for v in gates.values())
    report={"status":"success","decision":"AUTHORIZED_FOR_CONTROLLED_TRAINING" if approved else "NOT_AUTHORIZED_YET",
      "training_authorization":approved,
      "criteria":{"minimum_tokens_per_language":500,"maximum_loss_regression_vs_code_pct":5,
        "minimum_improvement_in_one_language_pct":2,"max_consecutive_identical_tokens_exclusive":8,
        "max_repeated_3gram_ratio_exclusive":0.75,"unknown_token_rate":0.0},
      "gates":gates,"corpus":corpora,"tokenizer":tokenizer_diag,
      "checkpoint":{"text_parameters":text.num_parameters(),"code_parameters":code.num_parameters(),
        "text_config":text.config.to_dict(),"code_config":code.config.to_dict()},
      "loss_perplexity_comparison":comparison,"generation":generations,
      "training_and_data_metadata":metadata,
      "interpretation":["A aprovação libera somente uma rodada controlada/experimental, não uma declaração de qualidade de produção.",
        "Corpus independente foi escrito para avaliação e não é coletado do corpus de treinamento.",
        "Loss e perplexidade são médias ponderadas por token; não são comparáveis a probes curtos anteriores.",
        "Geração usa amostragem fixa de parâmetros e diversidade é medida sobre tokens gerados."]}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
