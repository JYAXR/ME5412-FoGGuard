import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.sensor_variants import generate_variant


def input_frame():
    return pd.DataFrame({"timestamp": [0., .5, 1.], "subjectID": [1, 1, 1],
                         "ankleL_gyro_x": [1., np.nan, 2.], "ankleL_acc_x": [0., 1., 2.]})


def test_seeded_variants_preserve_identity_and_gaps():
    original = input_frame()
    a, params = generate_variant(original, 10)
    b, same = generate_variant(original, 10)
    c, _ = generate_variant(original, 11)
    pd.testing.assert_frame_equal(a, b)
    assert params == same and not a.equals(c)
    assert a.timestamp.equals(original.timestamp) and a.subjectID.equals(original.subjectID)
    assert a.ankleL_gyro_x.isna().equals(original.ankleL_gyro_x.isna())
    assert original.ankleL_acc_x.tolist() == [0., 1., 2.]


def test_labels_cannot_reach_generator():
    with pytest.raises(ValueError, match="labels"):
        generate_variant(input_frame().assign(fog=1), 10)
