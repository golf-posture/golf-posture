from pipeline.golfpose_joints import GOLFPOSE_JOINT_INDEX, golfpose_joints, predictions_payload


def keypoints() -> list[list[float]]:
    """17 points whose x is the index, so a wrong index shows up as a wrong x."""
    return [[float(i), 10.0 * i] for i in range(17)]


class TestGolfposeJoints:
    def test_uses_the_h36m_order_of_the_checkpoint_not_coco(self):
        # the checkpoint's keypoint_id2name: 11-13 left shoulder/elbow/wrist, 14-16 right
        assert GOLFPOSE_JOINT_INDEX["LEFT_SHOULDER"] == 11
        assert GOLFPOSE_JOINT_INDEX["RIGHT_SHOULDER"] == 14
        assert GOLFPOSE_JOINT_INDEX["LEFT_WRIST"] == 13
        assert GOLFPOSE_JOINT_INDEX["RIGHT_WRIST"] == 16

    def test_returns_benchmark_joints_with_the_heatmap_score_as_visibility(self):
        scores = [0.5] * 17
        scores[14] = 0.9

        joints = golfpose_joints(keypoints(), scores)

        assert joints["RIGHT_SHOULDER"] == (14.0, 140.0, 0.9, 0.0, 0.0, 0.0)
        assert joints["LEFT_ELBOW"][:3] == (12.0, 120.0, 0.5)


class TestPredictionsPayload:
    def test_has_the_layout_the_benchmark_cache_replays(self):
        payload = predictions_payload("golfpose", {"v.mp4": {3: {"A": (1.0, 2.0, 0.9, 0.0, 0.0, 0.0)}, 5: None}})

        assert payload["backend"] == "golfpose"
        assert payload["frames"]["v.mp4"]["3"] == {"A": [1.0, 2.0, 0.9, 0.0, 0.0, 0.0]}
        assert payload["frames"]["v.mp4"]["5"] is None
