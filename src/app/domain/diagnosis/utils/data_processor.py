# 진단 데이터 처리 유틸리티
import json
from typing import List, Any

def flatten_and_join(values: List[Any]) -> str:
    """
    Redis에서 가져온 데이터를 평탄화하고 문자열로 결합합니다.
    
    Args:
        values: Redis에서 가져온 데이터 리스트
        
    Returns:
        str: 평탄화되고 결합된 문자열
    """
    parts = []
    for v in values:
        if isinstance(v, dict):
            for val in v.values():
                if isinstance(val, (dict, list)):
                    parts.append(json.dumps(val, ensure_ascii=False))
                else:
                    parts.append(str(val))
        elif isinstance(v, list):
            parts.append(json.dumps(v, ensure_ascii=False))
        else:
            parts.append(str(v))
    return " ".join(parts)

def convert_to_json_string(data: Any) -> str:
    """
    딕셔너리나 리스트 형태의 데이터를 JSON 문자열로 변환합니다.
    
    Args:
        data: 변환할 데이터
        
    Returns:
        str: JSON 문자열로 변환된 데이터
    """
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    return str(data) 