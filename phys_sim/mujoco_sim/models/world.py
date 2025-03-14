import robosuite.macros as macros
from mujoco_sim.models.base import MujocoXML
from robosuite.utils.mjcf_utils import convert_to_string, find_elements, xml_path_completion


class MujocoWorldBase(MujocoXML):
    """Base class to inherit all mujoco worlds from."""

    def __init__(self):
        super().__init__(xml_path_completion("base.xml"))
        # Modify the simulation timestep to be the requested value
        options = find_elements(root=self.root, tags="option", attribs=None, return_first=True)
        options.set("timestep", convert_to_string(macros.SIMULATION_TIMESTEP))


class TwoFingerWorldBase(MujocoWorldBase):

    def __init__(self):
        super().__init__()
        self.merge_assets(MujocoXML(xml_path_completion("two_finger.xml")))
