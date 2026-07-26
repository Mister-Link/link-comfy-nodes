"""
FBX export for MotionStreamer 22-joint HumanML3D skeleton.
Uses FBX Python SDK to animate a template character FBX directly (no Blender needed).
Adapted from link-comfy-hymotion/hymotion/utils/smplh2woodfbx.py.
"""
import os
import shutil
import tempfile

import numpy as np

# Joint names for HumanML3D 22-joint skeleton (same ordering as SMPL-H body joints)
SMPL22_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]

# Lowercase aliases used in some FBX templates
_LOWERCASE = {
    "Pelvis": "pelvis", "L_Hip": "left_hip", "R_Hip": "right_hip",
    "Spine1": "spine1", "L_Knee": "left_knee", "R_Knee": "right_knee",
    "Spine2": "spine2", "L_Ankle": "left_ankle", "R_Ankle": "right_ankle",
    "Spine3": "spine3", "L_Foot": "left_foot", "R_Foot": "right_foot",
    "Neck": "neck", "L_Collar": "left_collar", "R_Collar": "right_collar",
    "Head": "head", "L_Shoulder": "left_shoulder", "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow", "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist", "R_Wrist": "right_wrist",
}


def write_fbx_with_character(
    template_fbx_path: str,
    rot_matrices: np.ndarray,
    translations: np.ndarray,
    save_path: str,
    fps: float = 30.0,
    scale: float = 100.0,
) -> bool:
    """
    Animate a rigged template FBX with 22-joint MotionStreamer output and save.

    Args:
        template_fbx_path: Path to rigged character template FBX.
        rot_matrices: (num_frames, 22, 3, 3) rotation matrices per joint.
        translations: (num_frames, 3) root translations in meters.
        save_path: Output FBX path.
        fps: Animation frame rate.
        scale: Translation scale (default 100 = meters to cm).

    Returns:
        True if save_path exists after export.
    """
    import fbx
    from transforms3d.euler import mat2euler

    translations_cm = translations * scale

    def _load_scene(mgr, filepath):
        imp = fbx.FbxImporter.Create(mgr, "")
        if not imp.Initialize(filepath, -1, mgr.GetIOSettings()):
            raise RuntimeError(f"FBX import failed: {imp.GetStatus().GetErrorString()}")
        sc = fbx.FbxScene.Create(mgr, "")
        imp.Import(sc)
        imp.Destroy()
        return sc

    def _collect_nodes(node, d=None):
        if d is None:
            d = {}
        d[node.GetName()] = node
        for i in range(node.GetChildCount()):
            _collect_nodes(node.GetChild(i), d)
        return d

    def _set_channel(layer, prop, axis, values, dt):
        idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        t = fbx.FbxTime()
        curve = prop.GetCurve(layer, axis, True)
        curve.KeyModifyBegin()
        for f, v in enumerate(values):
            t.SetSecondDouble(f * dt)
            ki = curve.KeyAdd(t)[0]
            curve.KeySetValue(ki, float(v[idx]))
            curve.KeySetInterpolation(ki, fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
        curve.KeyModifyEnd()

    def _animate_rotation(layer, node, rots_3x3, dt):
        eulers = []
        for r in rots_3x3:
            m = np.array(r, dtype=np.float64, copy=True)
            eulers.append(np.rad2deg(mat2euler(m, axes="sxyz")))
        for ax in ("X", "Y", "Z"):
            _set_channel(layer, node.LclRotation, ax, eulers, dt)

    def _animate_translation(layer, node, trans, dt):
        for ax in ("X", "Y", "Z"):
            _set_channel(layer, node.LclTranslation, ax, trans, dt)

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    try:
        scene = _load_scene(manager, template_fbx_path)

        mode = fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
        scene.GetGlobalSettings().SetTimeMode(mode)

        all_nodes = _collect_nodes(scene.GetRootNode())

        # Clear existing animations
        n_stacks = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
        for i in range(n_stacks - 1, -1, -1):
            s = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
            if s:
                s.Destroy()

        stack = fbx.FbxAnimStack.Create(scene, "MotionStreamer")
        layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)

        dt = 1.0 / fps
        root_applied = False

        for joint_idx, joint_name in enumerate(SMPL22_JOINT_NAMES):
            node = all_nodes.get(joint_name) or all_nodes.get(_LOWERCASE.get(joint_name, ""))
            if node is None:
                continue

            _animate_rotation(layer, node, rot_matrices[:, joint_idx], dt)

            if joint_idx == 0:  # Pelvis is root — also animate translation
                init_t = node.LclTranslation.Get()
                offset = np.array([init_t[0], init_t[1], init_t[2]])
                _animate_translation(layer, node, translations_cm + offset, dt)
                root_applied = True

        if not root_applied:
            print("[MotionStreamer FBX] Warning: Pelvis joint not found in template — no root translation applied")

        # Export into an isolated temp directory so the SDK can create whatever
        # sidecar files it likes (e.g. .fbm folders, .png textures) — the whole
        # directory is discarded after we copy just the .fbx out.
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = os.path.join(tmpdir, "export.fbx")
            ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
            ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
            ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
            exp = fbx.FbxExporter.Create(manager, "")
            if not exp.Initialize(tmp, -1, ios):
                raise RuntimeError(f"FBX export init failed: {exp.GetStatus().GetErrorString()}")
            exp.Export(scene)
            exp.Destroy()
            shutil.copy2(tmp, save_path)
            # tmpdir and all its contents (sidecars, .fbm dirs) are removed on exit

    finally:
        manager.Destroy()

    return os.path.exists(save_path)
