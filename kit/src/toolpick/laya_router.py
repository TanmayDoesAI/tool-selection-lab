"""Optional category router: the Laya decision model (convaiinnovations/laya-multilingual), run locally on CPU.
Install with: pip install 'toolpick[laya]'. Used by the `category-laya` strategy; no API calls, no tokens billed.
"""
import math

MODEL='convaiinnovations/laya-multilingual'
QUESTION="Which business area's tools are needed to fulfil this request?"   # the prompt used in the study

class LayaRouter:
    kind='laya'
    def __init__(self,categories,model=MODEL,threads=None):
        """categories: {name: one-line description}."""
        try:import laya,torch
        except ImportError as e:raise ImportError("category-laya needs the Laya model: pip install 'toolpick[laya]'") from e
        if threads:torch.set_num_threads(threads)
        self.categories=dict(categories);self.model=model;self.agent=laya.load(model,device='cpu');self._logits=None
        try:   # predict() rounds and saturates probabilities, so tap the raw logits for a usable top-2 ranking
            import laya.agent as la
            orig=la._option_logits
            def tap(logits,items,offset):
                rows=orig(logits,items,offset);self._logits=[[float(x) for x in (r.float().cpu().tolist() if hasattr(r,'float') else r)] for r in rows];return rows
            la._option_logits=tap
        except (ImportError,AttributeError):pass
    def route(self,request,k=2):
        """-> (top-k categories, {category: probability})."""
        self._logits=None
        r=self.agent.predict({'body':request},{'cat':{'type':'choice','instructions':QUESTION,'criteria':self.categories}})
        if self._logits:
            z=self._logits[0];m=max(z);e=[math.exp(x-m) for x in z];s=sum(e);probs={c:v/s for c,v in zip(self.categories,e)}
        else:
            ans=r['answers']['cat'];probs={c:float(ans.get('probabilities',{}).get(c,0.0)) for c in self.categories} if isinstance(ans,dict) else {}
            if not any(probs.values()):probs={c:float(c==(ans.get('choice') if isinstance(ans,dict) else ans)) for c in self.categories}
        return sorted(probs,key=lambda c:(-probs[c],c))[:k],probs

class KeywordRouter:
    """Offline stand-in for --dry-run and tests: ranks categories by the BM25 scores of their tools."""
    kind='keyword-router'
    def __init__(self,keyword_index,tool_categories):self.index=keyword_index;self.cat=tool_categories
    def route(self,request,k=2):
        probs={}
        for n,s in zip(self.index.names,self.index.scores(request)):probs[self.cat[n]]=max(probs.get(self.cat[n],0.0),s)
        return sorted(probs,key=lambda c:(-probs[c],c))[:k],probs
