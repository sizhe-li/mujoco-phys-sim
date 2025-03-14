import numpy as np

from robosuite.models.objects import MujocoXMLObject, MujocoGeneratedObject
from mujoco_sim.utils.io_utils import xml_path_completion

import xml.etree.ElementTree as ET


class MeshObject(MujocoXMLObject, MujocoGeneratedObject):
    """
    Generic mesh object
    """

    xml_path: str = "objects/letter_A.xml"

    def __init__(self, name, xml_path):
        self.xml_path = xml_path
        super().__init__(
            fname=xml_path_completion(self.xml_path),
            name=name,
            duplicate_collision_geoms=True,
        )
        self.size_info = self.get_size()

    def get_size(self):
        size_info = []
        # Parse the XML model of the object
        xml_tree = ET.ElementTree(ET.parse(xml_path_completion(self.xml_path)))

        # Find all geometries (geoms) that use a mesh
        for mesh in xml_tree.findall(".//mesh"):
            scale = np.fromstring(
                mesh.get("scale", "1 1 1"), sep=" "
            )  # Default scale is 1x1x1 if not specified
            size_info.append({"mesh": mesh, "scale": scale})
        return size_info

    @property
    def horizontal_radius(self):
        # TODO: calculate multi-mesh horizontal radius
        size = self.get_size()[0]["scale"]
        return np.linalg.norm(size[0:2], 2) + 1e-2

    @property
    def bottom_offset(self):
        # TODO: calculate multi-mesh size
        size = self.get_size()[0]["scale"]
        return np.array([0, 0, -1 * size[2]])

    @property
    def top_offset(self):
        # TODO: calculate multi-mesh size
        size = self.get_size()[0]["scale"]
        return np.array([0, 0, size[2]])


class LetterAObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_A.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterEObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_E.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterGObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_G.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterMObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_M.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterRObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_R.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterTObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_T.xml"

    def __init__(self, name):
        super().__init__(name=name)


class LetterVObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/letter_A.xml"

    def __init__(self, name):
        super().__init__(name=name)


class StarObject(MeshObject):
    """
    Letter A object
    """

    xml_path: str = "objects/star.xml"

    def __init__(self, name):
        super().__init__(name=name)
