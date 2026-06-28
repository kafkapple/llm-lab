import logging
import sys
import codecs
from pathlib import Path
from datetime import datetime
import shutil
def setup_logging(cfg, output_dir):
    """로깅 설정"""
    # 로거 생성
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 로그 디렉토리 생성
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 핸들러 설정
    log_file = logs_dir / 'run.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Windows 환경에서의 인코딩 설정
    if sys.platform == 'win32':
        try:
            import locale
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            pass
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    source_dir = Path(__file__).parent.parent / 'config'
    destination_dir = output_dir / 'config'
    print(f"Source: {source_dir}")
    print(f"Destination: {destination_dir}")

    shutil.copytree(source_dir, destination_dir)
    return logger

def safe_log(logger, level: str, message: str):
    """안전한 로깅을 위한 함수"""
    try:
        # 문자열이 아닌 경우 문자열로 변환
        if not isinstance(message, str):
            message = str(message)
        
        # 로그 레벨에 따른 로깅 함수 호출
        log_func = getattr(logger, level)
        log_func(message)
            
    except Exception as e:
        # 로깅 실패 시 기본 출력으로 폴백
        print(f"로깅 실패: {str(e)}")
        print(f"원본 메시지: {message}")
