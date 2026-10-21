"""Minimal end-to-end Streamlit demonstration."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from fogguard.cueing import AdaptiveCueingController
from fogguard.detection import FreezeIndexDetector
from fogguard.fogstar_pipeline import replay_fogstar_csv
from fogguard.pipeline import replay_csv

st.set_page_config(page_title="FoGGuard", page_icon="🚶", layout="wide")
st.title("FoGGuard——离线步态回放")
st.caption("仅用于课程项目，不用于诊断或临床使用。")

input_mode = st.sidebar.selectbox("输入管线", ("合成/标准 CSV", "FoG-STAR 多任务模型"))
default_path = (
    Path("data/raw/fogstar/sensor_data.csv")
    if input_mode == "FoG-STAR 多任务模型"
    else Path("data/demo/synthetic_gait.csv")
)
path_text = st.sidebar.text_input("回放 CSV", str(default_path))
model_path = st.sidebar.text_input(
    "FoG-STAR 模型", "artifacts/fogstar_multitask_baseline.joblib"
)
fog_threshold = st.sidebar.slider(
    "FoG 阈值", 0.0, 1.0, FreezeIndexDetector().threshold, 0.01,
    disabled=input_mode == "FoG-STAR 多任务模型",
)

if st.sidebar.button("开始回放", type="primary"):
    try:
        if input_mode == "FoG-STAR 多任务模型":
            result = replay_fogstar_csv(path_text, model_path=model_path)
            st.caption(
                "曲线为原始模型概率；event_phase 与 stable_fog_detected 是机器人 2 "
                "时序集成结果。真值只用于离线评估。"
            )
        else:
            detector = FreezeIndexDetector(threshold=fog_threshold)
            controller = AdaptiveCueingController(min_confidence=fog_threshold)
            result = replay_csv(path_text, detector=detector, controller=controller)
        st.plotly_chart(
            px.line(result, x="window_end_s", y="fog_probability"),
            width="stretch",
        )
        st.dataframe(result, width="stretch", hide_index=True)
    except Exception as exc:  # noqa: BLE001 - final UI boundary reports demo failures.
        st.error(f"回放失败：{exc}")
else:
    st.info("请先生成演示数据，再点击“开始回放”。")
