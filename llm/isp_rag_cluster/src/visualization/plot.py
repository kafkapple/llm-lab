import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import traceback
from .constants import (
    DEFAULT_DPI, DEFAULT_FIGSIZE, TITLE_SIZE, COLORS,
    FONT_SETTINGS, PLOT_CONFIG, METRIC_COLUMNS
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
digit = 1
col_name_model = 'model'
col_name_prompt = 'prompt'

def setup_plot_style(cfg, ax):
    """Plot 스타일 설정"""
    plt.style.use(cfg.plotting.style)
    for key, value in FONT_SETTINGS.items():
        plt.rcParams[key] = value

    # 그리드 설정
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    return ax


# @safe_process_dataframe
def plot_grouped_bar_chart(
    df_info: pd.DataFrame,
    csv_path: Union[str, Path],
    save_path: Union[str, Path],
    model_mapping: Optional[Dict[str, str]] = None,
    prompt_mapping: Optional[Dict[str, str]] = None,
    model_order: Optional[List[str]] = None,
    prompt_order: Optional[List[str]] = None,
    cfg: Optional[Dict[str, str]] = None
) -> None:
    """
    CSV 파일을 읽고, 각 모델별로 프롬프트 타입과 메트릭을 비교하는 grouped bar chart를 생성.
    """
    try:
        logger.info(f"\n============\nPlot Grouped Bar Chart\n============\n")

        df = pd.read_csv(csv_path)

        # 매핑 적용 전에 원본 저장
        df['original_model'] = df[col_name_model]
        df['original_prompt'] = df[col_name_prompt]
        
        # 매핑 적용
        if model_mapping:
            df[col_name_model] = df[col_name_model].map(lambda x: model_mapping.get(x, x))
        if prompt_mapping:
            df[col_name_prompt] = df[col_name_prompt].map(lambda x: prompt_mapping.get(x, x))
 
        # 필터링 및 정렬
        if model_order:
            ordered_models = [model_mapping.get(m, m) if model_mapping else m for m in model_order]
            df = df[df['original_model'].isin(model_order)]
            df[col_name_model] = pd.Categorical(df[col_name_model], categories= ordered_models, ordered=True)
            df = df.sort_values(col_name_model)
            models = df[col_name_model].cat.categories
        
        if prompt_order:
            template_names = [prompt_mapping.get(pt, pt) if prompt_mapping else pt for pt in prompt_order]
            df[col_name_prompt] = pd.Categorical(df[col_name_prompt], template_names, ordered=True)
        else:
            template_names = sorted(df[col_name_prompt].unique())
 
        # 메트릭 설정
        metrics = cfg.plotting.palette.metric.keys() #['precision', 'recall', 'f1-score']
        
        # subplot 레이아웃 계산
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        n_prompt = len(template_names)
        if n_models * n_prompt > 35:
            value_size = 4
        elif n_models * n_prompt > 11:
            value_size = 5 # cfg.plotting.font.value_size
        elif n_models * n_prompt > 5:
            value_size = 8
        else:
            value_size = 10
        # 그래프 생성
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
            
        # 각 모델별 서브플롯
        for idx, model in enumerate(models):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            x = np.arange(len(template_names))
            width = 0.25
            
            model_data = df[df[col_name_model] == model]
            
            for i, metric in enumerate(metrics):
                values = []
                for temp_type in template_names:
                    subset = model_data[model_data[col_name_prompt] == temp_type]
                    value = round(subset[metric].values[0] * 100, digit) if not subset.empty else 0
                    values.append(value)
                
                # 바 차트 그리기
                bars = ax.bar(x + (i-1)*width, values, width,
                            label=metric.replace('-', ' ').title(),
                            color=cfg.plotting.palette.metric[metric], alpha=cfg.plotting.options.alpha, edgecolor=cfg.plotting.options.edgecolor, linewidth=cfg.plotting.options.linewidth)
               
      
                # 값 레이블 표시
                for bar in bars:
                    height = bar.get_height()
                    if digit == 1:
                        bar_value = f'{height:.1f}'
                    elif digit == 2:
                        bar_value = f'{height:.2f}'
                    elif digit == 0:
                        bar_value = f'{height:.0f}'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           bar_value, ha='center', va='bottom', fontsize=value_size)
            
            # 축 설정
            ax.set_xticks(x)
            ax.set_xticklabels(template_names, rotation=45, ha='right')
            ax.set_ylabel("Score (%)")
            ax.set_title(model, pad=10, fontsize=cfg.plotting.font.title_size, fontweight='bold')
            ax.set_ylim(0, 110)
            ax = setup_plot_style(cfg, ax)

            
            # 첫 번째 subplot에만 범례 표시
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        # 남은 서브플롯 제거
        for row in range(n_rows):
            for col in range(n_cols):
                if row * n_cols + col >= len(models):
                    axes[row, col].remove()
        
        # 레이아웃 조절
        plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.0)
        
        # 저장
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        logger.info(f"Plot saved to {save_path}.png")
        
    except Exception as e:
        logger.error(f"Error in plot_grouped_bar_chart: {str(e)}")
        raise

def plot_unknown_analysis(df_unknown, save_path, model_mapping=None, prompt_mapping=None, 
                          model_order=None, prompt_order=None,
                          plot_config={
                              'heatmap': {'show_prompt': True},
                              'detail': {
                                  'x_axis': 'emotion',
                                  'group_by': col_name_prompt,
                                  'col_wrap': 3,
                                  'subplot_by': col_name_model
                              }
                          }, cfg: Optional[Dict[str, str]] = None):
    """
    unknown 분석 결과를 히트맵 및 상세 바 차트로 시각화.
    - 서브플롯의 y축 스케일은 전체 데이터에서 최대값을 기준으로 동일하게 설정.
    - model_order, prompt_order 에 포함된 항목만 필터링하여 사용.
    - 색상 팔레트를 개선해 가시성을 높임.
    """
    logger.info(f"\n============\nplot_unknown_analysis\n============\n")
    print(cfg.plotting.style)
    plt.style.use(cfg.plotting.style)
    
    # 매핑 적용
    if model_mapping:
        df_unknown[col_name_model] = df_unknown[col_name_model].map(lambda x: model_mapping.get(x, x))
    if prompt_mapping:
        df_unknown[col_name_prompt] = df_unknown[col_name_prompt].map(lambda x: prompt_mapping.get(x, x))

    # 데이터 준비: unknown_ratio 관련 컬럼들 추출
    emotion_cols = [col for col in df_unknown.columns if col.endswith('_unknown_ratio') and col != 'total_unknown_ratio']
    emotion_data = df_unknown.melt(
        id_vars=[col_name_model, col_name_prompt, 'timestamp'],
        value_vars=emotion_cols,
        var_name='emotion',
        value_name='unknown_ratio'
    )
    emotion_data['emotion'] = emotion_data['emotion'].str.replace('_unknown_ratio', '')
    
    # model_order, prompt_order에 포함된 항목만 필터링 (매핑 후의 이름 사용)
    if model_order:
        allowed_models = [model_mapping.get(m, m) if model_mapping else m for m in model_order]
        emotion_data = emotion_data[emotion_data[col_name_model].isin(allowed_models)]
    if prompt_order:
        allowed_prompts = [prompt_mapping.get(p, p) if prompt_mapping else p for p in prompt_order]
        emotion_data = emotion_data[emotion_data[col_name_prompt].isin(allowed_prompts)]

    # Categorical 순서 설정
    if model_order:
        emotion_data[col_name_model] = pd.Categorical(
            emotion_data[col_name_model],
            [model_mapping.get(m, m) if model_mapping else m for m in model_order],
            ordered=True
        )
    if prompt_order:
        emotion_data[col_name_prompt] = pd.Categorical(
            emotion_data[col_name_prompt],
            [prompt_mapping.get(p, p) if prompt_mapping else p for p in prompt_order],
            ordered=True
        )

    # 전체 unknown_ratio의 최대값 계산 (y축 스케일 통일용)
    global_max = emotion_data['unknown_ratio'].max()
    y_limit = (global_max * 1.1) if global_max > 0 else 100  # 여유 10% 추가

    # 상세 바 차트: 세련된 색상 팔레트 적용 (예: YlOrRd 계열)
    palette = sns.color_palette(cfg.plotting.palette.unknown, n_colors=len(emotion_data[col_name_prompt].unique()))
    
    config = plot_config['detail']
    unique_values = emotion_data[config['subplot_by']].unique()
    n_subplots = len(unique_values)
    
    # n_subplots가 0인 경우 처리
    if n_subplots == 0:
        logger.warning("No data to plot after filtering")
        return
        
    # col_wrap이 0이거나 n_subplots보다 큰 경우 처리
    col_wrap = max(1, min(config['col_wrap'], n_subplots))
    n_cols = min(col_wrap, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols

    # 서브플롯 생성 및 동일한 y축 범위 적용
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    for idx, subplot_val in enumerate(unique_values, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        data = emotion_data[emotion_data[config['subplot_by']] == subplot_val]
        
        # barplot 생성: x축은 'emotion', 그룹은 'prompt'
        sns.barplot(data=data, x=config['x_axis'], y='unknown_ratio', hue=config['group_by'],
                    ax=ax, palette=palette)
        
        ax.set_title(f"{subplot_val}", fontsize=cfg.plotting.font.title_size, fontweight='bold')
        ax.set_xlabel(config['x_axis'].title(), fontsize=cfg.plotting.font.label_size)
        ax.set_ylabel('Unknown Ratio (%)', fontsize=cfg.plotting.font.label_size)
        ax.set_ylim(0, 30)  # 모든 서브플롯 동일한 y축
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45)
        
        # 범례는 첫번째 서브플롯에만 표시
        if idx == 1:
            ax.legend(title=config['group_by'].title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.legend().remove()
    
    plt.tight_layout()
    save_path = Path(save_path)
    directory  = save_path.parent
    filename = save_path.name
    save_path = directory / f"fig_bar_{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Detailed unknown analysis saved to {save_path}")


def plot_rag_search_analysis(df_rag, save_path, model_mapping=None, prompt_mapping=None, embedding_mapping=None,
                             model_order=None, prompt_order=None, embedding_order=None,
                             plot_config={
                                 'heatmap': {'show_prompt': True},
                                 'detail': {
                                     'x_axis': 'emotion',
                                     'group_by': col_name_prompt,
                                     'col_wrap': 3,
                                     'subplot_by': col_name_model
                                 }
                             }):
    """
    RAG 검색 결과를 히트맵 및 상세 바 차트(매치/불일치 비교)로 시각화.
    """
    logger.info(f"\n============\nplot_rag_search_analysis\n============\n")
    
    # 데이터 디버그 출력
    logger.info(f"Initial df_rag shape: {df_rag.shape}")
    logger.info(f"df_rag columns: {df_rag.columns.tolist()}")
    
    plt.style.use('seaborn-v0_8-paper')
    if model_mapping:
        df_rag[col_name_model] = df_rag[col_name_model].map(lambda x: model_mapping.get(x, x))
    if prompt_mapping:
        df_rag[col_name_prompt] = df_rag[col_name_prompt].map(lambda x: prompt_mapping.get(x, x))

    # RAG 프롬프트만 필터링
    df_rag = df_rag[df_rag[col_name_prompt].str.contains('RAG', case=False, na=False)]
    
    # Match Ratio 시각화
    colors_match = ['#2ecc71', '#27ae60', '#1e8449']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors_match)
    emotion_cols_match = [col for col in df_rag.columns if col.endswith('_match_ratio') and col != 'total_match_ratio']
    emotion_data_match = df_rag.melt(
        id_vars=[col_name_model, col_name_prompt],
        value_vars=emotion_cols_match,
        var_name='emotion',
        value_name='match_ratio'
    )
    emotion_data_match['emotion'] = emotion_data_match['emotion'].str.replace('_match_ratio', '')
    
    # model_order, prompt_order에 포함된 항목만 필터링
    if model_order:
        allowed_models = [model_mapping.get(m, m) if model_mapping else m for m in model_order]
        emotion_data_match = emotion_data_match[emotion_data_match[col_name_model].isin(allowed_models)]
    if prompt_order:
        allowed_prompts = [prompt_mapping.get(p, p) if prompt_mapping else p for p in prompt_order]
        emotion_data_match = emotion_data_match[emotion_data_match[col_name_prompt].isin(allowed_prompts)]

    # Categorical 순서 설정
    if model_order:
        emotion_data_match[col_name_model] = pd.Categorical(
            emotion_data_match[col_name_model],
            [model_mapping.get(m, m) if model_mapping else m for m in model_order],
            ordered=True
        )
    if prompt_order:
        emotion_data_match[col_name_prompt] = pd.Categorical(
            emotion_data_match[col_name_prompt],
            [prompt_mapping.get(p, p) if prompt_mapping else p for p in prompt_order],
            ordered=True
        )
    
    # plt.figure(figsize=(15, 10))
    # if plot_config['heatmap']['show_prompt']:
    pivot_data = emotion_data_match.pivot_table(
        values='match_ratio',
        index=[col_name_model, col_name_prompt],
        columns='emotion',
        aggfunc='mean',
        observed=True  # 실제 존재하는 조합만 사용
    )
    # else:
    #     pivot_data = emotion_data_match.pivot_table(
    #         values='match_ratio',
    #         index=col_name_model,
    #         columns='emotion',
    #         aggfunc='mean',
    #         observed=True  # 실제 존재하는 조합만 사용
    #     )
    
    # 실제 존재하는 모델과 프롬프트만 필터링
    if model_order:
        existing_models = pivot_data.index.get_level_values(col_name_model).unique() if plot_config['heatmap']['show_prompt'] else pivot_data.index
        ordered_models = [m for m in [model_mapping.get(m, m) if model_mapping else m for m in model_order] if m in existing_models]
        
        
        # if plot_config['heatmap']['show_prompt']:
        existing_prompts = pivot_data.index.get_level_values(col_name_prompt).unique()
        ordered_prompts = [p for p in [prompt_mapping.get(p, p) if prompt_mapping else p for p in prompt_order] if p in existing_prompts]
        new_index = pd.MultiIndex.from_product([ordered_models, ordered_prompts], names=[col_name_model, col_name_prompt])
        pivot_data = pivot_data.reindex(new_index).dropna(how='all')
        # else:
        #     pivot_data = pivot_data.reindex(ordered_models).dropna(how='all')
    
    # Mismatch Ratio도 동일한 방식으로 처리
    emotion_cols_mismatch = [col for col in df_rag.columns if col.endswith('_mismatch_ratio') and col != 'total_mismatch_ratio']
    emotion_data_mismatch = df_rag.melt(
        id_vars=[col_name_model, col_name_prompt],
        value_vars=emotion_cols_mismatch,
        var_name='emotion',
        value_name='mismatch_ratio'
    )
    emotion_data_mismatch['emotion'] = emotion_data_mismatch['emotion'].str.replace('_mismatch_ratio', '')
    
    # model_order, prompt_order에 포함된 항목만 필터링
    if model_order:
        emotion_data_mismatch = emotion_data_mismatch[emotion_data_mismatch[col_name_model].isin(allowed_models)]
    if prompt_order:
        emotion_data_mismatch = emotion_data_mismatch[emotion_data_mismatch[col_name_prompt].isin(allowed_prompts)]
    
    # Categorical 순서 설정
    if model_order:
        emotion_data_mismatch[col_name_model] = pd.Categorical(
            emotion_data_mismatch[col_name_model],
            ordered_models,
            ordered=True
        )
    if prompt_order:
        emotion_data_mismatch[col_name_prompt] = pd.Categorical(
            emotion_data_mismatch[col_name_prompt],
            ordered_prompts if 'ordered_prompts' in locals() else allowed_prompts,
            ordered=True
        )

    # Mismatch Ratio 시각화
    colors_mismatch = ['#e74c3c', '#c0392b', '#962e23']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors_mismatch)
    
    
    # 상세 그래프 (Match/Mismatch 비교)
    config = plot_config['detail']
    unique_values = emotion_data_match[config['subplot_by']].unique()
    n_subplots = len(unique_values)
    
    # n_subplots가 0인 경우 처리
    if n_subplots == 0:
        logger.warning("No data to plot after filtering")
        return
        
    # col_wrap이 0이거나 n_subplots보다 큰 경우 처리
    col_wrap = max(1, min(config['col_wrap'], n_subplots))
    n_cols = min(col_wrap, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    for idx, subplot_val in enumerate(unique_values, 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)
        data_match = emotion_data_match[emotion_data_match[config['subplot_by']] == subplot_val]
        data_mismatch = emotion_data_mismatch[emotion_data_mismatch[config['subplot_by']] == subplot_val]
        
        print(f"\nPlotting subplot {idx} for {subplot_val}")
        print("data_match shape:", data_match.shape)
        print("data_mismatch shape:", data_mismatch.shape)
        
        # barplot 생성 시 label 파라미터 제거 (legend는 나중에 따로 설정)
        sns.barplot(data=data_match, x=config['x_axis'], y='match_ratio', hue=config['group_by'],
                    ax=ax, palette=colors_match, alpha=0.7)
        ax2 = ax.twinx()
        sns.barplot(data=data_mismatch, x=config['x_axis'], y='mismatch_ratio', hue=config['group_by'],
                    ax=ax2, palette=colors_mismatch, alpha=0.3)
        
        ax.set_title(f"{subplot_val}")
        ax.set_xlabel(config['x_axis'].title())
        ax.set_ylabel('Match Ratio (%)', color=colors_match[0])
        ax2.set_ylabel('Mismatch Ratio (%)', color=colors_mismatch[0])
        
        # 범례 설정 수정
        if idx == 1:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            print(f"Legend labels - Match: {labels1}, Mismatch: {labels2}")
            
            # 범례 생성 전에 기존 범례 제거
            ax.legend().remove()
            ax2.legend().remove()
            
            # 새로운 범례 생성
            ax.legend(title=config['group_by'].title(), 
                     handles=lines1 + lines2,
                     labels=['Match', 'Mismatch'],
                     bbox_to_anchor=(1.05, 1), 
                     loc='upper left')
        else:
            ax.legend().remove()
            ax2.legend().remove()
        
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_metrics_comparison(
    df: pd.DataFrame,
    save_path: Union[str, Path],
    model_mapping: Optional[Dict[str, str]] = None,
    prompt_mapping: Optional[Dict[str, str]] = None,
    model_order: Optional[List[str]] = None,
    prompt_order: Optional[List[str]] = None,
    cfg: Optional[Dict[str, str]] = None,
) -> None:
    """각 감정 클래스별 precision, recall, f1-score를 비교하는 그래프 생성
       template 컬럼이 있을 경우, 각 template별 stde 값으로 scatter error bar 추가
    """
    try:
        logger.info(f"\n============\nPlot Class Metrics Comparison\n============\n")

        # 1. 데이터 확인
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
        
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return
            
        # 2. 매핑 적용
        if model_mapping:
            df['Model'] = df['Model'].map(lambda x: model_mapping.get(x, x))
        if prompt_mapping:
            df['Prompt'] = df['Prompt'].map(lambda x: prompt_mapping.get(x, x))

        logger.info(f"Unique models after mapping: {df['Model'].unique()}")
        logger.info(f"Unique prompts after mapping: {df['Prompt'].unique()}")
        # 3. 모델과 프롬프트 순서 적용
        if model_order:
            df = df[df['Model'].isin(model_order)]
        if prompt_order:
            df = df[df['Prompt'].isin(prompt_order)]

        if df.empty:
            logger.warning("DataFrame is empty after filtering")
            return
        
        # 4. 전체 Value의 최대값 계산 (y축 스케일 통일용)
        global_max = df['Value'].max()
        y_limit = (global_max * 1.1) if global_max > 0 else 1.0  # 여유 10% 추가
        
        # 5. 서브플롯 설정
        if model_order:
            ordered_models = [model_mapping.get(m, m) if model_mapping else m for m in model_order]
            df = df[df['Model'].isin(ordered_models)]
            df['Model'] = pd.Categorical(df['Model'], categories= ordered_models, ordered=True)
            df = df.sort_values('Model')
            models = df['Model'].cat.categories
        n_models = len(models)
        
        if n_models == 0:
            logger.warning("No models found after filtering")
            return
            
        n_cols = min(3, n_models)  # 최대 3열
        n_rows = (n_models + n_cols - 1) // n_cols
        # 6. 서브플롯 생성
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        # 고정 색상 팔레트: precision, recall, f1 (예: 초록 ~ 파랑 계열)
        emotions = cfg.data.datasets[cfg.data.name].labels
        metrics = cfg.plotting.palette.metric
        
        # 7. 각 모델별 서브플롯 생성
        for idx, model in enumerate(models, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx)
            model_data = df[df['Model'] == model]
            x = np.arange(len(emotions))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                # 각 메트릭별로 해당하는 감정의 값만 추출
                metric_data = model_data[model_data['Metric'] == metric]
                values = []
                for j, emotion in enumerate(emotions):
                    # 기존 방식: 해당 감정에 대한 첫 번째 값을 사용
                    value = metric_data[metric_data['Emotion'] == emotion]['Value'].mean() if not metric_data[metric_data['Emotion'] == emotion].empty else 0
                    values.append(value)        #.iloc[0]
                # bar plot 그리기
                ax.bar(x + i * width, values, width, label=metric, color=cfg.plotting.palette.metric[metric],
                       alpha=cfg.plotting.options.alpha, edgecolor=cfg.plotting.options.edgecolor, linewidth=cfg.plotting.options.linewidth)
                # (Emotion, Metric) 그룹 전체의 평균 및 standard error (Template 무관)
                for j, emotion in enumerate(emotions):
                    overall_rows = metric_data[metric_data['Emotion'] == emotion]
                    if not overall_rows.empty:
                        overall_mean = overall_rows['Value'].mean()
                        n_overall = len(overall_rows)
                        overall_stde = overall_rows['Value'].std(ddof=1) / np.sqrt(n_overall) if n_overall > 1 else 0
                        # 중앙 위치: bar의 중심 (x[j] + i*width)
                        central_x = x[j] + i * width
                        ax.errorbar(central_x, overall_mean, yerr=overall_stde, fmt='none',
                                    elinewidth=0.5, capthick=0.5, capsize=1, zorder=2,
                                    color=cfg.plotting.palette.error)
                # 7.1. template 컬럼이 존재하면, 각 template별로 scatter error bar 추가
                if 'Prompt' in metric_data.columns:
                    templates = sorted(metric_data['Prompt'].unique())
                    n_templates = len(templates)
                    # 각 감정에 대해 template별 scatter 점 추가
                    for j, emotion in enumerate(emotions):
                        for t_idx, template in enumerate(templates):
                            # 각 template에 대해 bar 내 오프셋 조정
                            scatter_offset = (t_idx - (n_templates - 1) / 2) * (width / (n_templates + 1))
                            scatter_x = x[j] + i * width + scatter_offset
                            # (Emotion, Metric, Prompt) 조합으로 해당 row 필터링
                            row = metric_data[(metric_data['Emotion'] == emotion) & (metric_data['Prompt'] == template)]
                            if not row.empty:
                                scatter_val = row['Value'].mean()
                                # n = len(row)
                                # stde_val = row['Value'].std(ddof=1) / np.sqrt(n) if n > 1 else 0
                                ax.scatter(scatter_x, scatter_val, marker='o', s=10, alpha=0.6,
                                        color=cfg.plotting.palette.metric[metric], zorder=4)
                
            # 7.2. 그래프 꾸미기
            ax.set_title(f"{model}", fontsize=cfg.plotting.font.title_size, fontweight='bold')
            ax.set_xlabel('Emotion', fontsize=cfg.plotting.font.label_size)
            ax.set_ylabel('Score (%)', fontsize=cfg.plotting.font.label_size)
            ax.set_ylim(0, y_limit)  # 모든 서브플롯 동일한 y축
            ax.set_xticks(x + width)
            ax.set_xticklabels(emotions, rotation=45)
            
            # 7.3. 범례는 첫번째 서브플롯에만 표시
            if idx == 1:
                ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend().remove()
            ax = setup_plot_style(cfg, ax)
        
        # 8. 그래프 저장
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error in plot_class_metrics_comparison: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
