"""Custom data config for the RoboCasa365 PandaOmron checkpoint.

Loadable via load_data_config("structrl.robocasa_data_config:RobocasaPandaOmronConfig").

Differs from gr00t's stock SinglePandaGripperDataConfig only in the video_keys —
the ckpt was trained on `video.robot0_*` keys (matching the LeRobot dataset
column names), not `video.left_view/...`.
"""
from gr00t.experiment.data_config import SinglePandaGripperDataConfig


class RobocasaPandaOmronConfig(SinglePandaGripperDataConfig):
    # MUST match keys present in the ckpt's experiment_cfg/metadata.json,
    # which are the LeRobot dataset key names with a `video.` prefix.
    # The gym wrapper outputs different names (video.res256_image_side_0 etc.)
    # so the simulation client has to remap obs keys before sending to server.
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
