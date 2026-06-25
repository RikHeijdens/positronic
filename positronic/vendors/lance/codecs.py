"""Lance codecs.

Mirror the existing LeRobot codecs but add static scalar fields that the Lance
converter writes as scalar columns in the wide per-episode table.
"""

import uuid as _uuid

import configuronic as cfn

from positronic.cfg import codecs as base
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, Get
from positronic.policy.codec import Codec
from positronic.policy.observation import ObservationCodec


def _random_uuid(_episode: Episode) -> str:
    return str(_uuid.uuid4())


class _StaticScalars(Codec):
    """Adds arbitrary static scalars derived from the raw episode."""

    def __init__(self, **derivations):
        self._derivations = derivations

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Derive(**self._derivations)


@cfn.config(fps=15.0, horizon=None, binarize_grip=None, uuid=False)
def _compose(obs, action, fps: float, horizon: float | None, binarize_grip, uuid: bool):
    derivations = {'current_task': Get('task', ''), 'language_instruction1': Get('task', '')}
    if uuid:
        derivations['uuid'] = _random_uuid
    inner = base.compose(obs=obs, action=action, fps=fps, horizon=horizon, binarize_grip=binarize_grip)
    return inner & _StaticScalars(**derivations)


ee = _compose.override(obs=base.eepose_obs.override(image_size=(512, 512)), action=base.absolute_pos_action)


@cfn.config(image_size=(512, 512))
def ee_joints_obs(image_size):
    """Observation encoder emitting EE-pose and joint-position state in delineated columns."""
    return ObservationCodec(
        state={
            'observation.state_ee': {'robot_state.ee_pose': 7, 'grip': 1},
            'observation.state_joints': {'robot_state.q': 7, 'grip': 1},
        },
        images={
            'observation.images.left': ('image.wrist', tuple(image_size)),
            'observation.images.side': ('image.exterior', tuple(image_size)),
        },
    )


@cfn.config(
    action_ee=base.absolute_pos_action.override(action_key='action_ee'),
    action_joints=base.ik_joints_action.override(action_key='action_joints', solver='lm'),
)
def ee_joints_action(action_ee, action_joints):
    """Absolute EE-pose action and IK-reconstructed joint action in delineated columns."""
    return action_ee & action_joints


# Single-pass codec writing both representations: EE columns (`*_ee`) come from the recorded
# end-effector pose/command, joint columns (`*_joints`) from the recorded joint state and an
# IK reconstruction of the commanded pose. Aligned by construction (one resampling grid).
ee_joints = _compose.override(obs=ee_joints_obs, action=ee_joints_action, uuid=True)
