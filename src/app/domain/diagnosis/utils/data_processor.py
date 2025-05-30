# 진단 데이터 처리 유틸리티
import json
from typing import List, Any
import logging

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

def extract_top_classification(classification_data: Any) -> str:
    """
    classification 데이터에서 첫 번째 클래스와 확률값을 추출합니다.
    
    Args:
        classification_data: classification 데이터 (문자열 또는 딕셔너리)
        
    Returns:
        str: "[Computer Vision-based Skin Disease Diagnosis System analysis result]:" 형식의 문자열
    """
    try:
        if not classification_data:
            logging.warning("classification_data가 비어있습니다.")
            return ""
            
        # 딕셔너리인 경우
        if isinstance(classification_data, dict):
            if 'result' in classification_data and classification_data['result']:
                first_result = classification_data['result'][0]
                if 'class' in first_result and 'probability' in first_result:
                    class_name = first_result['class']
                    probability = first_result['probability']
                    # 확률값을 퍼센트로 변환하고 소수점 1자리까지 표시
                    probability_percent = round(probability * 100, 1)
                    result = f"**[Computer Vision-based Skin Disease Diagnosis System analysis result]:** {probability_percent}% probability of {class_name}."
                    return result
            logging.warning("딕셔너리에서 클래스를 찾을 수 없습니다.")
            return ""
            
        logging.warning(f"지원하지 않는 데이터 타입입니다: {type(classification_data)}")
        return ""
    except Exception as e:
        logging.error(f"클래스 추출 중 오류 발생: {str(e)}")
        return "" 