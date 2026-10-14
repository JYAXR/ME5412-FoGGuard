"""Minimal end-to-end Streamlit demonstration."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fogguard.cueing import AdaptiveCueingController
from fogguard.data import WindowReplay, load_csv
from fogguard.detection import FreezeIndexDetector
from fogguard.turning import YawRateTurnRecognizer
from fogguard.types import ControllerContext

st.set_page_config(page_title="FoGGuard", page_icon="🚶", layout="wide")
st.title("FoGGuard — offline gait replay")
st.caption("Coursework prototype only; not for diagnosis or clinical use.")

default_path = Path("data/demo/synthetic_gait.csv")
path_text = st.sidebar.text_input("Replay CSV", str(default_path))
fog_threshold = st.sidebar.slider("FoG threshold", 0.0, 1.0, 0.60, 0.05)

if st.sidebar.button("Run replay", type="primary"):
    try:
        frame = load_csv(path_text)
        replay = WindowReplay(frame)
        detector = FreezeIndexDetector(threshold=fog_threshold)
        turner = YawRateTurnRecognizer()
        controller = AdaptiveCueingController(min_confidence=fog_threshold)
        rows = []
        for window in replay.windows():
            fog = detector.predict(window)
            turn = turner.predict(window)
            cue = controller.decide(ControllerContext(float(window.timestamp_s[-1]), fog, turn))
            rows.append(
                {
                    "time_s": float(window.timestamp_s[-1]),
                    "fog_probability": fog.probability,
                    "turn": turn.direction.value,
                    "cue": cue.cue_type.value,
                }
            )
        result = pd.DataFrame(rows)
        st.plotly_chart(px.line(result, x="time_s", y="fog_probability"), use_container_width=True)
        st.dataframe(result, use_container_width=True, hide_index=True)
    except Exception as exc:  # Keep the demo failure visible to the team.
        st.error(f"Replay failed: {exc}")
else:
    st.info("Generate demo data, then choose Run replay.")
