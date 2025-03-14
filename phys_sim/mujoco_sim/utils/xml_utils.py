import mujoco


def load_mj_model_from_xml(xml_file, scale=None):
    if scale is not None and isinstance(scale, float):
        from dm_control import mjcf

        mjcf_model = mjcf.from_path(xml_file)

        kwrds = ["mesh", "geom", "body"]
        attrs = ["scale", "size", "pos"]

        for k in kwrds:
            instances = mjcf_model.find_all(k)
            for inst in instances:
                for a in attrs:
                    if hasattr(inst, a):
                        val = inst.__getattr__(a)
                        if val is not None:
                            val *= scale

        xml_str = mjcf_model.to_xml_string()
        assets = dict()

        for mjcf_asset in mjcf_model.find_all("mesh"):
            asset_file = mjcf_asset._get_attribute("file").get_vfs_filename()
            assets[asset_file] = mjcf_asset._get_attribute("file").contents

        model = mujoco.MjModel.from_xml_string(xml_str, assets)

    else:
        model = mujoco.MjModel.from_xml_path(xml_file)

    return model
