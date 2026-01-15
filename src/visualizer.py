import sys
# 确保你的 src 文件夹在路径里
sys.path.append('/content/drive/MyDrive/CA6001_DEPRESSION_PROJECT/src')

# 导入你的主题文件
import theme

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import theme  # <--- 【关键】导入你的主题模块

# --- 辅助函数：将 HEX 颜色转为 RGBA 字符串 ---
def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {opacity})'

def plot_radar_chart(df_raw):
    """
    接收原始 DataFrame，执行清洗、映射、归一化并绘制雷达图。
    (已更新配色方案，引用 theme.py)
    """
    # 1. 复制一份以免修改原数据
    df_viz = df_raw.copy()

    # 2. 字符串清洗
    obj_cols = df_viz.select_dtypes(include=['object']).columns
    for col in obj_cols: df_viz[col] = df_viz[col].str.strip()

    # 3. 映射逻辑
    sleep_map = {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 2}
    diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3, 'Others': 2}
    df_viz['Sleep Duration'] = df_viz['Sleep Duration'].map(sleep_map).fillna(2)
    df_viz['Dietary Habits'] = df_viz['Dietary Habits'].map(diet_map).fillna(2)

    df_viz['CGPA_Cleaned'] = df_viz['CGPA'].replace(0, np.nan)
    df_viz['Study Satisfaction_Cleaned'] = df_viz['Study Satisfaction'].replace(0, np.nan)
    df_viz['Financial Stress'] = df_viz['Financial Stress'].replace(0, np.nan)

    # Age Clip (15-35)
    age_min_view = 15
    age_max_view = 35
    df_viz['Age_Cleaned'] = df_viz['Age'].apply(lambda x: np.nan if x < 10 else x)
    df_viz['Age_View'] = df_viz['Age_Cleaned'].clip(lower=age_min_view, upper=age_max_view)

    # 4. 标签定义
    labels_with_range = [
        f'Age<br><span style="color:grey;font-size:10px">({age_min_view}-{age_max_view})</span>',
        'Sleep Duration<br><span style="color:grey;font-size:10px">(1-4 Scale)</span>',
        'Dietary Habits<br><span style="color:grey;font-size:10px">(1-3 Scale)</span>',
        'CGPA<br><span style="color:grey;font-size:10px">(5-10 Scale)</span>',
        'Study Satisfaction<br><span style="color:grey;font-size:10px">(1-5 Scale)</span>',
        'Academic Pressure<br><span style="color:grey;font-size:10px">(1-5 Scale)</span>',
        'Financial Stress<br><span style="color:grey;font-size:10px">(1-5 Scale)</span>'
    ]

    features_to_plot = [
        'Age_View', 'Sleep Duration', 'Dietary Habits', 'CGPA_Cleaned',
        'Study Satisfaction_Cleaned', 'Academic Pressure', 'Financial Stress'
    ]

    # 5. 自定义归一化
    feature_ranges = {
        'Age_View': [15, 35], 'Sleep Duration': [0, 5], 'Dietary Habits': [0, 4],
        'CGPA_Cleaned': [4, 10], 'Study Satisfaction_Cleaned': [0, 6],
        'Academic Pressure': [0, 6], 'Financial Stress': [0, 6]
    }

    scaler = MinMaxScaler()
    df_scaled = df_viz.copy()

    for col in features_to_plot:
        min_v, max_v = feature_ranges[col]
        df_scaled[col] = (df_viz[col] - min_v) / (max_v - min_v)

    # 6. 统计计算
    def get_stats(df_group):
        means = df_group[features_to_plot].mean().tolist()
        q_upper = df_group[features_to_plot].quantile(0.75).tolist()
        q_lower = df_group[features_to_plot].quantile(0.25).tolist()
        return means, q_upper, q_lower

    mean_h, q_upper_h, q_lower_h = get_stats(df_scaled[df_scaled['Depression'] == 0])
    mean_d, q_upper_d, q_lower_d = get_stats(df_scaled[df_scaled['Depression'] == 1])

    # 7. 绘图
    fig = go.Figure()
    categories = labels_with_range + [labels_with_range[0]]

    def add_ribbon(fig, q_lower, q_upper, color_fill, name):
        fig.add_trace(go.Scatterpolar(
            r=q_upper + [q_upper[0]], theta=categories, mode='lines', line_width=0, showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatterpolar(
            r=q_lower + [q_lower[0]], theta=categories, mode='lines', line_width=0,
            fill='tonext', fillcolor=color_fill, showlegend=True,
            name=f'{name} (25%-75% Range)'
        ))

    # ======================================================
    # 【核心修改】引用 theme.py 中的颜色
    # ======================================================
    
    # 1. Healthy Group (使用 theme.COLOR_PROTECTIVE)
    # 填充色：将 HEX 转为 RGBA，透明度 0.3
    fill_h = hex_to_rgba(theme.COLOR_PROTECTIVE, 0.3)
    # 线条色：直接使用 HEX
    line_h = theme.COLOR_PROTECTIVE
    
    add_ribbon(fig, q_lower_h, q_upper_h, fill_h, 'Healthy')
    fig.add_trace(go.Scatterpolar(
        r=mean_h + [mean_h[0]], theta=categories, mode='lines+markers', name='Healthy Mean',
        line=dict(color=line_h, width=2), marker=dict(size=5)
    ))

    # 2. Depressed Group (使用 theme.COLOR_RISK)
    # 填充色：将 HEX 转为 RGBA，透明度 0.3
    fill_d = hex_to_rgba(theme.COLOR_RISK, 0.3)
    # 线条色：直接使用 HEX
    line_d = theme.COLOR_RISK
    
    add_ribbon(fig, q_lower_d, q_upper_d, fill_d, 'Depressed')
    fig.add_trace(go.Scatterpolar(
        r=mean_d + [mean_d[0]], theta=categories, mode='lines+markers', name='Depressed Mean',
        line=dict(color=line_d, width=3), marker=dict(size=6)
    ))

    # 布局更新
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='lightgrey'),
            angularaxis=dict(tickfont=dict(size=12, color='black'), rotation=0)
        ),
        title=dict(
            text=(
                '<b>Feature Profile Comparison (Normalized)</b><br>'
                '<span style="font-size:13px;color:grey;font-weight:normal">'
                'Outer edge indicates higher values / greater intensity'
                '</span>'
            ),
            y=0.95,
            x=0.05,
            xanchor='left',
            yanchor='top',
            font=dict(size=20)
        ),
        legend=dict(
            y=1.2,
            x=1.2,
            yanchor="top",
            xanchor="right",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11)
        ),
        template='plotly_white',
        width=900, height=750,
        margin=dict(t=120, b=50, l=80, r=80)
    )

    return fig
