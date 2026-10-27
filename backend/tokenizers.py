"""Tokenizer inspection with no repository-provided Python execution."""
from pathlib import Path
import hashlib
import json
import shutil
import os

# This service only loads tokenizers, never neural models. Avoid importing ML runtimes.
os.environ.setdefault('USE_TORCH','0')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')

ALLOWED_SUFFIXES={'.json','.model','.txt','.tiktoken','.vocab','.merges'}


class TokenizerService:
    def __init__(self, root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.loaded={}

    def load(self, source, revision=None):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        source=str(source)
        path=Path(source).expanduser()
        if path.exists():
            if path.is_file() and path.suffix == '.json':
                tok=PreTrainedTokenizerFast(tokenizer_file=str(path))
                # tokenizer.json alone does not describe the semantic roles of all special tokens.
            elif path.is_file() and path.suffix == '.model':
                from transformers import LlamaTokenizer
                tok=LlamaTokenizer(vocab_file=str(path),legacy=True)
            else:
                tok=AutoTokenizer.from_pretrained(str(path),trust_remote_code=False,local_files_only=True)
        else:
            tok=AutoTokenizer.from_pretrained(source,revision=revision,trust_remote_code=False)
        token_id=hashlib.sha256((source+'@'+str(revision)).encode()).hexdigest()[:20]
        target=self.root/token_id;target.mkdir(exist_ok=True)
        tok.save_pretrained(target)
        self.loaded[token_id]=tok
        vocab=tok.get_vocab();ids=list(vocab.values())
        info={'id':token_id,'source':source,'repo':None if path.exists() else source,'revision':revision,'size':len(vocab),'capacity':max(ids)+1 if ids else 0,'base_size':tok.vocab_size,'added_tokens':tok.get_added_vocab(),'special_tokens':{k:{'token':str(v),'id':tok.convert_tokens_to_ids(str(v))} for k,v in tok.special_tokens_map.items() if not isinstance(v,list)},'mask_id':tok.mask_token_id,'pad_id':tok.pad_token_id,'max_length':tok.model_max_length if tok.model_max_length < 10**9 else None,'class':type(tok).__name__}
        (target/'studio_metadata.json').write_text(json.dumps(info))
        return info

    def preview(self, token_id, text):
        from transformers import AutoTokenizer
        if token_id not in self.loaded:
            path=self.root/token_id
            if not path.is_dir():raise ValueError('Tokenizer is not available on this server. Load it again.')
            self.loaded[token_id]=AutoTokenizer.from_pretrained(path,trust_remote_code=False,local_files_only=True)
        tok=self.loaded[token_id]; result=tok(text,add_special_tokens=True)
        return {'ids':result['input_ids'],'tokens':tok.convert_ids_to_tokens(result['input_ids']),'shape':[1,len(result['input_ids'])],'attention_mask':result.get('attention_mask')}
