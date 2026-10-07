"""MuJoCo model and world-state export for the native display viewer.

The field list matches ``MujocoParser.ts`` and ``mujocoParserHelpers.ts``.
Keep MuJoCo-specific names and conversion here, outside the display transport.
"""

import json

import numpy as np

MODEL_COUNTS = ("nbody", "nflex", "ngeom", "nhfield", "nlight", "ntendon")
MODEL_ARRAYS = (
    "flex_vertadr", "flex_vertnum", "geom_bodyid", "geom_conaffinity",
    "geom_contype", "geom_dataid", "geom_group", "geom_matid",
    "geom_pos", "geom_quat", "geom_rgba", "geom_size", "geom_type",
    "hfield_adr", "hfield_data", "hfield_ncol", "hfield_nrow",
    "hfield_size", "light_attenuation", "light_castshadow",
    "light_diffuse", "mat_reflectance", "mat_rgba",
    "mat_shininess", "mat_specular", "mat_texrepeat",
    "mesh_face", "mesh_faceadr", "mesh_facenormal", "mesh_facenum",
    "mesh_facetexcoord", "mesh_normal", "mesh_normaladr",
    "mesh_normalnum", "mesh_texcoord", "mesh_texcoordadr",
    "mesh_texcoordnum", "mesh_vert", "mesh_vertadr", "mesh_vertnum",
    "name_bodyadr", "names", "tendon_width", "tex_adr",
    "tex_height", "tex_pathadr", "tex_type", "tex_width",
)
GEOM_TYPES = (
    "mjGEOM_PLANE", "mjGEOM_HFIELD", "mjGEOM_SPHERE", "mjGEOM_CAPSULE",
    "mjGEOM_ELLIPSOID", "mjGEOM_CYLINDER", "mjGEOM_BOX", "mjGEOM_MESH",
)
FRAME_ARRAYS = (
    "xpos", "xquat", "light_xpos", "light_xdir", "flexvert_xpos",
    "ten_wrapadr", "ten_wrapnum", "wrap_xpos", "hfield_data",
)


def _flat_numbers(value, label):
    # Modern native MuJoCo exposes names as bytes, whereas WASM exposes the
    # same NUL-delimited UTF-8 bytes as a Uint8Array.
    if isinstance(value, bytes):
        return list(value)
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"[display] {label} is not a numeric MuJoCo array")
    return array.reshape(-1).tolist()


class MujocoSceneProvider:
    def __init__(self, model, data, mujoco_module, project):
        self._model = model
        self._data = data
        self._mujoco = mujoco_module
        # mj_step integrates qpos after its last kinematics pass. Recompute
        # display positions in separate data so the controller's MjData is
        # untouched and the scene frame depicts the integrated state.
        self._view_data = mujoco_module.MjData(model)
        self.frame_arrays = tuple(
            np.asarray(
                model.hfield_data if name == "hfield_data" else getattr(self._view_data, name)
            )
            for name in FRAME_ARRAYS
        )
        self.frame_size = 4 * (1 + sum(array.size for array in self.frame_arrays))
        model_fields = {name: int(getattr(model, name)) for name in MODEL_COUNTS}
        model_fields.update(
            (name, _flat_numbers(getattr(model, name), name)) for name in MODEL_ARRAYS
        )
        # MuJoCo 3.2 stores RGB-only textures and six material slots; newer
        # versions store variable-channel tex_data and ten texture roles. The
        # browser parser consumes the newer layout in either case.
        if hasattr(model, "tex_data"):
            model_fields["tex_data"] = _flat_numbers(model.tex_data, "tex_data")
            model_fields["tex_nchannel"] = _flat_numbers(model.tex_nchannel, "tex_nchannel")
        else:
            model_fields["tex_data"] = _flat_numbers(model.tex_rgb, "tex_rgb")
            model_fields["tex_nchannel"] = [3] * len(model.tex_type)
        if hasattr(model, "light_type"):
            model_fields["light_type"] = _flat_numbers(model.light_type, "light_type")
        else:
            model_fields["light_type"] = [
                1 if directional else 0 for directional in model.light_directional
            ]
        material_texture_ids = np.asarray(model.mat_texid)
        if material_texture_ids.shape == (model.nmat, 10):
            model_fields["mat_texid"] = _flat_numbers(material_texture_ids, "mat_texid")
        elif material_texture_ids.shape == (model.nmat, 6):
            roles = np.full((model.nmat, 10), -1, dtype=np.int32)
            roles[:, 1] = material_texture_ids[:, 0]
            model_fields["mat_texid"] = roles.reshape(-1).tolist()
        else:
            raise ValueError(
                f"[display] unsupported mat_texid shape {material_texture_ids.shape}"
            )
        message = {
            "type": "scene",
            "version": 1,
            "project": project,
            "model": model_fields,
            "geomTypes": {
                name: int(getattr(mujoco_module.mjtGeom, name)) for name in GEOM_TYPES
            },
            "frameFields": [
                {"name": name, "size": int(array.size)}
                for name, array in zip(FRAME_ARRAYS, self.frame_arrays)
            ],
        }
        self.payload = json.dumps(message, allow_nan=False, separators=(",", ":"))

    def frame(self, time_seconds):
        # Run synchronously on the stepping thread. The native data's xpos and
        # related visual fields are stale immediately after mj_step integrates.
        np.copyto(self._view_data.qpos, self._data.qpos)
        np.copyto(self._view_data.mocap_pos, self._data.mocap_pos)
        np.copyto(self._view_data.mocap_quat, self._data.mocap_quat)
        self._mujoco.mj_kinematics(self._model, self._view_data)
        self._mujoco.mj_camlight(self._model, self._view_data)
        self._mujoco.mj_flex(self._model, self._view_data)
        self._mujoco.mj_tendon(self._model, self._view_data)
        # Own the snapshot before socket I/O begins.
        values = np.empty(1 + sum(array.size for array in self.frame_arrays), dtype="<f4")
        values[0] = time_seconds
        offset = 1
        for array in self.frame_arrays:
            end = offset + array.size
            values[offset:end] = array.reshape(-1)
            offset = end
        return values.tobytes()
