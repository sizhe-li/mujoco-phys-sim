from mujoco_sim.models.arenas import Arena
from mujoco_sim.utils.io_utils import xml_path_completion


class EmptyArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
