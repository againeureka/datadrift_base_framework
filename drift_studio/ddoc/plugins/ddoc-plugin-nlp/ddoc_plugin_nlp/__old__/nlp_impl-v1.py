# plugins/ddoc-plugin-nlp/ddoc_plugin_nlp/nlp_impl.py
import pluggy
from typing import Any, Dict
from ddoc.plugins.hookspecs import HookSpecs

hookimpl = pluggy.HookimplMarker("ddoc")

class DDOCNlpPlugin:
    DDOC_HOOKSPEC_MIN = "1.0.0"
    DDOC_HOOKSPEC_MAX = "1.999.999"
        
    @hookimpl
    def transform_apply(self, input_path: str, transform: str, args: Dict[str, Any], output_path: str):
        """
        Provide NLP transforms: text.tokenize, text.normalize (toy examples).
        """
        if transform not in {"text.tokenize", "text.normalize"}:
            return None  # not handled by this plugin
    
        # minimal toy behavior
        from pathlib import Path
        text = Path(input_path).read_text(encoding="utf-8")
    
        if transform == "text.tokenize":
            tokens = text.split()  # naive
            Path(output_path).write_text("\n".join(tokens), encoding="utf-8")
        elif transform == "text.normalize":
            norm = " ".join(text.split())
            Path(output_path).write_text(norm, encoding="utf-8")
    
        return {"ok": True, "written": output_path, "transform": transform, "provider": "ddoc_nlp"}
