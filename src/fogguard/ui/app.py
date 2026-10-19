"""Minimal end-to-end Streamlit demonstration."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from fogguard.cueing import AdaptiveCueingController
from fogguard.detection import FreezeIndexDetector
from fogguard.pipeline import replay_csv

st.set_page_config(page_title="FoGGuard", page_icon="🚶", layout="wide")
st.title("FoGGuard——离线步态回放")
st.caption("仅用于课程项目，不用于诊断或临床使用。")

default_path = Path("data/demo/synthetic_gait.csv")
path_text = st.sidebar.text_input("回放 CSV", str(default_path))
fog_threshold = st.sidebar.slider("FoG 阈值", 0.0, 1.0, FreezeIndexDetector().threshold, 0.01)

if st.sidebar.button("开始回放", type="primary"):
    try:
        detector = FreezeIndexDetector(threshold=fog_threshold)
        controller = AdaptiveCueingController(min_confidence=fog_threshold)
        result = replay_csv(path_text, detector=detector, controller=controller)
        st.plotly_chart(
            px.line(result, x="window_end_s", y="fog_probability"),
            use_container_width=True,
        )
        st.dataframe(result, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001 - final UI boundary reports demo failures.
        st.error(f"回放失败：{exc}")
else:
    st.info("请先生成演示数据，再点击“开始回放”。")
