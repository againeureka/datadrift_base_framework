# ddoc/plugins/ddoc-plugin-nlp/nlp_impl.py
import pluggy
from typing import Any, Dict, Optional, List

# pluggy 마커 정의 (ddoc/core/plugins.py에서 GROUP="ddoc"으로 정의했으므로 일치)
hookimpl = pluggy.HookimplMarker("ddoc")

class DDOCNlpPlugin:
    """
    DDOC NLP 플러그인은 텍스트 데이터에 대한 토큰화 및 정규화 변환을 제공합니다.
    """
    # ddoc 핵심 시스템과의 호환성을 위한 훅스펙 버전 범위 정의
    DDOC_HOOKSPEC_MIN = "1.0.0"
    DDOC_HOOKSPEC_MAX = "1.999.999"
    
    @hookimpl(tryfirst=True) # 변환 훅을 우선적으로 실행하도록 설정
    def transform_apply(self, input_path: str, transform: str, args: Dict[str, Any], output_path: str):
        """
        NLP 변환: text.tokenize, text.normalize (예시 구현).
        """
        if transform not in {"text.tokenize", "text.normalize"}:
            return None  # 이 플러그인에서 처리하지 않는 변환은 None 반환
    
        # 최소한의 토이(toy) 동작 구현
        from pathlib import Path
        try:
            text = Path(input_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            # 실패 시 명확한 딕셔너리 반환 (None이 아니므로 다음 플러그인으로 처리가 넘어가지 않음)
            return {"ok": False, "error": f"Input file not found: {input_path}", "provider": "ddoc_nlp"}
        except Exception as e:
            return {"ok": False, "error": f"Error reading input file: {e}", "provider": "ddoc_nlp"}

        if transform == "text.tokenize":
            tokens = text.split()  # 단순 토큰화
            Path(output_path).write_text("\n".join(tokens), encoding="utf-8")
        elif transform == "text.normalize":
            norm = " ".join(text.split()) # 다중 공백 제거
            Path(output_path).write_text(norm, encoding="utf-8")
    
        # 훅이 성공적으로 처리했음을 나타내는 결과 반환
        return {"ok": True, "written": output_path, "transform": transform, "provider": "ddoc_nlp"}

    # @hookimpl(hookwrapper=True) 제거, 단순 구현 훅으로 변경
    @hookimpl
    def ddoc_get_metadata(self) -> Dict[str, Any]:
        """
        플러그인의 메타데이터를 제공하는 훅 (설명 기능).
        이 훅이 ddoc/plugins/hookspecs.py에 정의되어 있다고 가정합니다.
        """
        # 이 플러그인의 메타데이터를 반환합니다.
        return {
            "name": "ddoc_nlp",
            "version": "1.0.0",
            "author": "Gemini",
            "description": "텍스트 데이터의 토큰화 및 정규화와 같은 기본 NLP 전처리 기능을 제공합니다.",
            "hooks_implemented": [
                "transform_apply (text.tokenize, text.normalize)",
                "ddoc_get_metadata"
            ]
        }


