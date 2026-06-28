from typing import Dict, List

# 시각화 관련 상수
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (15, 10)
TITLE_SIZE = 14

# 색상 팔레트
COLORS = {
    'match': ['#2ecc71', '#27ae60', '#1e8449'],
    'mismatch': ['#e74c3c', '#c0392b', '#962e23']
}

# 폰트 설정
FONT_SETTINGS = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12
}

# 플롯 설정
PLOT_CONFIG = {
    'heatmap': {'show_prompt': True},
    'detail': {
        'x_axis': 'emotion',
        'group_by': 'prompt',
        'col_wrap': 3,
        'subplot_by': 'model'
    }
}

# 메트릭 컬럼
METRIC_COLUMNS = ['precision', 'recall', 'f1-score']

# 파일 패턴
FILE_PATTERNS = {
    'classification_report': 'classification_report_*.csv',
    'dataset': 'dataset-*.csv',
    'rag_search': '*rag_search_results*.csv'
} 