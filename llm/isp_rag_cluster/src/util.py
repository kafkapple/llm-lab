import re
from datetime import datetime
import os
import argparse
import pathlib

import sys

import requests
def get_chat_id(bot_token):
    url = f'https://api.telegram.org/bot{bot_token}/getUpdates'
    response = requests.get(url)
    data = response.json()
    
    for result in data['result']:
        chat_id = result['message']['chat']['id']
        print(f"Chat ID: {chat_id}")
from telegram import Bot
async def send_telegram_message(message, bot_token, chat_id):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)


    # asyncio.run()으로 비동기 함수 실행

class Logger:
    def __init__(self, filename):
        # 기존 표준출력 객체 저장 (터미널)
        self.terminal = sys.stdout
        # 로그 파일 열기 (append 모드)
        self.log = open(filename, "a", encoding="utf-8")
    
    def write(self, message):
        # 터미널에 출력
        self.terminal.write(message)
        # 로그 파일에도 기록
        self.log.write(message)
    
    def flush(self):
        # 파이썬의 버퍼 flush를 지원하기 위해 구현 (예: print의 flush=True)
        self.terminal.flush()
        self.log.flush()
def enable_logging(log_filename="print_log.txt"):
    sys.stdout = Logger(log_filename)

from transformers import AutoTokenizer

def count_tokens(prompt: str, model: str) -> int:
    # Load the model-specific tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return len(tokens)


def sanitize_model_name(model: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model.lower())


def get_output_dir_name(model_name: str, cfg) -> str:
    """Create output folder name based on settings"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [timestamp, model_name]
 
    template_name = str(cfg.model.template)
    if isinstance(template_name, str) and template_name.startswith("${prompt."):
        # Extract the actual template name from the reference
        template_name = template_name.split("${prompt.")[1].rstrip("}")
    name_parts.append(template_name)
    if "rag" in cfg.model.template:
        sanitized_model_name = sanitize_model_name(cfg.rag.embedding_model)
        name_parts.append(sanitized_model_name)
    
    return "_".join(name_parts)


def remove_mac_files(directory, dry_run=False, force=False):
    """
    지정한 디렉터리와 그 하위 디렉터리에서 "._"로 시작하는 파일들을 찾아 삭제합니다.
    
    매번 "정말 삭제할 것인가?" 라는 의문을 품을 수밖에 없는 작업인데요.
    
    :param directory: 검색을 시작할 루트 디렉터리
    :param dry_run: True이면 삭제하지 않고 어떤 파일들이 삭제될지 출력만 함.
    :param force: True이면 사용자 확인 없이 바로 삭제.
    :return: 삭제된(또는 dry_run 모드에서는 삭제될) 파일 경로들의 리스트.
    """
    to_remove = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                file_path = os.path.join(root, file)
                to_remove.append(file_path)
                
    if not to_remove:
        print("삭제할 '._' 파일이 없습니다. 정말로 불필요한 파일인가요?")
        return []
    
    print(f"총 {len(to_remove)}개의 '._' 파일을 찾았습니다.")
    
    if dry_run:
        print("dry-run 모드: 아래의 파일들이 삭제될 예정입니다.")
        for f in to_remove:
            print(f)
        return to_remove
    
    if not force:
        response = input("정말로 이 파일들을 영구 삭제하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("삭제 작업이 취소되었습니다. 정말 삭제해도 되는지 다시 한 번 생각해보세요.")
            return []
    
    removed_files = []
    for file_path in to_remove:
        try:
            os.remove(file_path)
            print(f"삭제됨: {file_path}")
            removed_files.append(file_path)
        except Exception as e:
            print(f"삭제 실패 - {file_path}: {e}")
            
    return removed_files


def main():
    parser = argparse.ArgumentParser(
        description="macOS에서 생성된 '._' 파일들을 찾아 삭제하는 모듈입니다."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../outputs/done_4",
        help="검색할 디렉터리 경로"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 삭제하지 않고, 삭제될 파일들만 출력"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="삭제 전 사용자 확인 없이 바로 진행"
    )
    args = parser.parse_args()
    
    removed_files = remove_mac_files(args.directory, dry_run=args.dry_run, force=args.force)
    
    if args.dry_run:
        print("dry-run 모드 종료: 실제 파일은 삭제되지 않았습니다.")
    elif removed_files:
        print(f"총 {len(removed_files)}개의 파일이 삭제되었습니다.")
    else:
        print("삭제된 파일이 없습니다.")


def rename_folders_with_pathlib(root_dir, find_pattern, replace_pattern):
    """
    pathlib을 사용하여 'rag_prompt'로 끝나는 폴더명을 찾아
    해당 폴더명 뒤에 '_'를 추가합니다.
    
    Parameters:
    root_dir (str): 검색할 루트 디렉토리 경로
    
    Returns:
    list: 변경된 폴더 목록 (원래 이름, 새 이름)
    """
    root_path = pathlib.Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}'는 유효한 디렉토리가 아닙니다.")
    
    renamed_folders = []
    pattern = re.compile(rf'{find_pattern}$')
    
    # 모든 디렉토리 검색
    for path in root_path.iterdir():
        if pattern.search(path.name):
            new_name = f"{path.name}{replace_pattern}"
            new_path = path.parent / new_name
            
            try:
                path.rename(new_path)
                renamed_folders.append((str(path), str(new_path)))
                print(f"폴더 이름 변경: '{path}' -> '{new_path}'")
            except Exception as e:
                print(f"폴더 이름 변경 중 오류 발생: '{path}' -> '{new_path}': {e}")
    
    return renamed_folders


def rename_folder():
    root_dir ='results/isear'

    print(f"'{root_dir}' 디렉토리에서 'rag_prompt'로 끝나는 폴더를 검색 중...")
  
    renamed = rename_folders_with_pathlib(root_dir, 'rag_prompt', 'rag_prompt_j-hartmann_emotion-english-distilroberta-base')
 
    
    if renamed:
        print("\n변경된 폴더 목록:")
        for old, new in renamed:
            print(f"{old} -> {new}")
    else:
        print("\n변경된 폴더가 없습니다.")


if __name__ == "__main__":
    #main()
    rename_folder()
