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

# Parent index for each joint in SMPL22_JOINT_NAMES (-1 = root). Standard SMPL/
# HumanML3D 22-joint kinematic tree.
SMPL22_PARENTS = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8,
    9, 9, 9, 12,
    13, 14,
    16, 17,
    18, 19,
]

# Which outgoing segment each joint should visually point along when exported as
# a standalone skeleton. Branch joints still need a single display axis.
SMPL22_PRIMARY_CHILD = {
    0: 3,
    1: 4,
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 9,
    7: 10,
    8: 11,
    9: 12,
    12: 15,
    13: 16,
    14: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
}

# face_joint_idx convention used throughout HumanML3D: r_hip, l_hip, r_shoulder, l_shoulder
_R_HIP, _L_HIP, _R_SHOULDER, _L_SHOULDER = 2, 1, 17, 16

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


def _load_scene(fbx_mod, mgr, filepath):
    imp = fbx_mod.FbxImporter.Create(mgr, "")
    if not imp.Initialize(filepath, -1, mgr.GetIOSettings()):
        raise RuntimeError(f"FBX import failed: {imp.GetStatus().GetErrorString()}")
    sc = fbx_mod.FbxScene.Create(mgr, "")
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


def _find_node(all_nodes, joint_name):
    return all_nodes.get(joint_name) or all_nodes.get(_LOWERCASE.get(joint_name, ""))


def _set_channel(fbx_mod, layer, prop, axis, values, dt):
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    t = fbx_mod.FbxTime()
    curve = prop.GetCurve(layer, axis, True)
    curve.KeyModifyBegin()
    for f, v in enumerate(values):
        t.SetSecondDouble(f * dt)
        ki = curve.KeyAdd(t)[0]
        curve.KeySetValue(ki, float(v[idx]))
        curve.KeySetInterpolation(ki, fbx_mod.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant)
    curve.KeyModifyEnd()


def _animate_rotation(fbx_mod, mat2euler, layer, node, rots_3x3, dt):
    eulers = []
    for r in rots_3x3:
        m = np.array(r, dtype=np.float64, copy=True)
        eulers.append(np.rad2deg(mat2euler(m, axes="sxyz")))
    for ax in ("X", "Y", "Z"):
        _set_channel(fbx_mod, layer, node.LclRotation, ax, eulers, dt)


def _animate_translation(fbx_mod, layer, node, trans, dt):
    for ax in ("X", "Y", "Z"):
        _set_channel(fbx_mod, layer, node.LclTranslation, ax, trans, dt)


def _clear_anim_stacks(fbx_mod, scene):
    n_stacks = scene.GetSrcObjectCount(fbx_mod.FbxCriteria.ObjectType(fbx_mod.FbxAnimStack.ClassId))
    for i in range(n_stacks - 1, -1, -1):
        s = scene.GetSrcObject(fbx_mod.FbxCriteria.ObjectType(fbx_mod.FbxAnimStack.ClassId), i)
        if s:
            s.Destroy()


def _export_scene(fbx_mod, manager, ios, scene, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "export.fbx")
        ios.SetBoolProp(fbx_mod.EXP_FBX_EMBEDDED, True)
        ios.SetBoolProp(fbx_mod.EXP_FBX_MATERIAL, True)
        ios.SetBoolProp(fbx_mod.EXP_FBX_TEXTURE, True)
        exp = fbx_mod.FbxExporter.Create(manager, "")
        if not exp.Initialize(tmp, -1, ios):
            raise RuntimeError(f"FBX export init failed: {exp.GetStatus().GetErrorString()}")
        exp.Export(scene)
        exp.Destroy()
        shutil.copy2(tmp, save_path)
        # tmpdir and all its contents (sidecars, .fbm dirs) are removed on exit


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

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    try:
        scene = _load_scene(fbx, manager, template_fbx_path)

        mode = fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
        scene.GetGlobalSettings().SetTimeMode(mode)

        all_nodes = _collect_nodes(scene.GetRootNode())
        _clear_anim_stacks(fbx, scene)

        stack = fbx.FbxAnimStack.Create(scene, "MotionStreamer")
        layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)

        dt = 1.0 / fps
        root_applied = False

        for joint_idx, joint_name in enumerate(SMPL22_JOINT_NAMES):
            node = _find_node(all_nodes, joint_name)
            if node is None:
                continue

            _animate_rotation(fbx, mat2euler, layer, node, rot_matrices[:, joint_idx], dt)

            if joint_idx == 0:  # Pelvis is root — also animate translation
                # translations_cm is already the absolute world position of the
                # root joint (matches the previewer's motion.xyz[:, 0]); Pelvis's
                # parent ("Reference") sits at identity, so local == world here.
                _animate_translation(fbx, layer, node, translations_cm, dt)
                root_applied = True

        if not root_applied:
            print("[MotionStreamer FBX] Warning: Pelvis joint not found in template — no root translation applied")

        _export_scene(fbx, manager, ios, scene, save_path)

    finally:
        manager.Destroy()

    return os.path.exists(save_path)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal-angle 3x3 rotation matrix R such that R @ a_hat == b_hat (Rodrigues' formula).

    Twist about the a/b axis itself is left undetermined (identity) — endpoint
    positions alone cannot recover bone roll, only swing.
    """
    a_hat = a / (np.linalg.norm(a) + 1e-12)
    b_hat = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a_hat, b_hat)
    s = np.linalg.norm(v)
    c = float(np.dot(a_hat, b_hat))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180-degree flip: rotate about any axis perpendicular to a_hat
        perp = np.array([1.0, 0.0, 0.0]) if abs(a_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a_hat, perp)
        axis /= np.linalg.norm(axis)
        return _rotation_about_axis(axis, np.pi)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def _extract_twist(r_full: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Swing-twist decompose r_full about `axis` (unit vector) and return only
    the twist (roll) component, as a 3x3 matrix. This is the rotation about the
    bone's own long axis that two endpoint positions alone can never reveal."""
    from transforms3d.quaternions import mat2quat, quat2mat

    q = mat2quat(np.asarray(r_full, dtype=np.float64))  # [w, x, y, z]
    proj = float(np.dot(q[1:], axis))
    q_twist = np.array([q[0], proj * axis[0], proj * axis[1], proj * axis[2]])
    n = np.linalg.norm(q_twist)
    if n < 1e-8:
        return np.eye(3)
    return quat2mat(q_twist / n)


def _normalize(vec: np.ndarray, fallback: "np.ndarray | None" = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return vec / norm
    if fallback is None:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-8:
        return fallback / fallback_norm
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _basis_from_y(y_axis: np.ndarray, guide: np.ndarray) -> np.ndarray:
    y = _normalize(y_axis)
    guide_proj = guide - y * float(np.dot(guide, y))
    if np.linalg.norm(guide_proj) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, y))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        guide_proj = fallback - y * float(np.dot(fallback, y))
    z = _normalize(np.cross(guide_proj, y), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    x = _normalize(np.cross(y, z), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    z = _normalize(np.cross(x, y), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return np.column_stack([x, y, z])


def _root_guide(pos: np.ndarray) -> np.ndarray:
    hips = pos[_R_HIP] - pos[_L_HIP]
    spine = pos[3] - pos[0]
    guide = np.cross(hips, spine)
    return _normalize(guide, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))


def _display_direction(pos: np.ndarray, joint_idx: int) -> np.ndarray:
    child = SMPL22_PRIMARY_CHILD.get(joint_idx)
    if child is not None:
        return pos[child] - pos[joint_idx]
    parent = SMPL22_PARENTS[joint_idx]
    if parent >= 0:
        return pos[joint_idx] - pos[parent]
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _build_world_bases(pos: np.ndarray) -> list[np.ndarray]:
    bases: list[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in SMPL22_JOINT_NAMES]
    for joint_idx in range(len(SMPL22_JOINT_NAMES)):
        y_axis = _display_direction(pos, joint_idx)
        if joint_idx == 0:
            guide = _root_guide(pos)
        else:
            guide = bases[SMPL22_PARENTS[joint_idx]][:, 2]
        bases[joint_idx] = _basis_from_y(y_axis, guide)
    return bases


def _create_clean_skeleton_scene(fbx_mod, manager, fps: float):
    scene = fbx_mod.FbxScene.Create(manager, "MotionStreamer")
    mode = fbx_mod.FbxTime().ConvertFrameRateToTimeMode(fps)
    scene.GetGlobalSettings().SetTimeMode(mode)

    scene_root = scene.GetRootNode()
    ref_node = fbx_mod.FbxNode.Create(scene, "Reference")
    ref_node.SetNodeAttribute(fbx_mod.FbxNull.Create(scene, "ReferenceNull"))
    scene_root.AddChild(ref_node)

    joint_nodes = []
    for joint_idx, joint_name in enumerate(SMPL22_JOINT_NAMES):
        skel_attr = fbx_mod.FbxSkeleton.Create(scene, joint_name)
        skel_attr.SetSkeletonType(fbx_mod.FbxSkeleton.EType.eLimbNode)
        try:
            skel_attr.Size.Set(1.0)
        except Exception:
            pass
        node = fbx_mod.FbxNode.Create(scene, joint_name)
        node.SetNodeAttribute(skel_attr)
        if SMPL22_PARENTS[joint_idx] == -1:
            ref_node.AddChild(node)
        else:
            joint_nodes[SMPL22_PARENTS[joint_idx]].AddChild(node)
        joint_nodes.append(node)

    return scene, joint_nodes


def write_fbx_from_positions(
    template_fbx_path: str,
    xyz: np.ndarray,
    save_path: str,
    fps: float = 30.0,
    scale: float = 100.0,
    rot_matrices: "np.ndarray | None" = None,
    source_name: str = "MotionStreamer",
    source_key: str = "MOTIONSTREAMER",
) -> bool:
    """
    Export a clean standalone FBX skeleton directly from 22-joint world-space
    positions (HumanML3D SMPL22 order, meters, Y-up).

    The earlier template-driven exporters preserved the skinned character's
    hidden rig basis, which is fine for driving that mesh but produces warped
    armatures in consumers that reconstruct visible bones from the FBX skeleton
    itself (Blender, NLF Studio). This path writes a fresh skeleton whose joint
    translations and local bases are derived directly from `xyz`, so the armature
    lines up with the same pose the 3D preview renders.

    Args:
        template_fbx_path: Unused compatibility placeholder.
        xyz: (num_frames, 22, 3) joint positions in meters, SMPL22 order.
        save_path: Output FBX path.
        fps: Animation frame rate.
        scale: Position scale (default 100 = meters to cm).
        rot_matrices: Unused compatibility placeholder.
        source_name: Human-readable source name stored in FBX metadata.
        source_key: Stable NLF source identifier stored in FBX metadata.

    Returns:
        True if save_path exists after export.
    """
    import fbx
    from transforms3d.euler import mat2euler

    _ = template_fbx_path, rot_matrices

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (len(SMPL22_JOINT_NAMES), 3):
        raise ValueError(
            f"Expected xyz with shape (num_frames, {len(SMPL22_JOINT_NAMES)}, 3), got {xyz.shape}"
        )

    num_frames = int(xyz.shape[0])
    num_joints = len(SMPL22_JOINT_NAMES)

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    try:
        scene, joint_nodes = _create_clean_skeleton_scene(fbx, manager, fps)

        # Keep the source identity inside the FBX so consumers do not have to
        # infer it from a filename or from the shared 22-joint template.
        info = fbx.FbxDocumentInfo.Create(manager, "NLF_Export_Metadata")
        info.mTitle = fbx.FbxString(source_name)
        info.mSubject = fbx.FbxString(f"NLF_SOURCE={source_key}")
        info.mKeywords = fbx.FbxString(f"NLF;{source_name};{source_key}")
        info.mComment = fbx.FbxString(f"NLF Studio source: {source_name}")
        scene.SetDocumentInfo(info)

        stack = fbx.FbxAnimStack.Create(scene, "MotionStreamer")
        layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)
        dt = 1.0 / fps

        local_eulers = [[] for _ in range(num_joints)]
        local_translations = [[] for _ in range(num_joints)]

        for f in range(num_frames):
            pos_cm = xyz[f] * scale
            world_basis = _build_world_bases(pos_cm)

            for j, node in enumerate(joint_nodes):
                parent = SMPL22_PARENTS[j]
                if parent == -1:
                    local_t = pos_cm[j]
                    local_r = world_basis[j]
                else:
                    parent_basis = world_basis[parent]
                    local_t = parent_basis.T @ (pos_cm[j] - pos_cm[parent])
                    local_r = parent_basis.T @ world_basis[j]
                euler_deg = np.rad2deg(mat2euler(local_r, axes="sxyz"))
                local_translations[j].append(local_t)
                local_eulers[j].append(euler_deg)
                if f == 0:
                    node.LclTranslation.Set(fbx.FbxDouble3(*local_t))
                    node.LclRotation.Set(fbx.FbxDouble3(*euler_deg))

        for j, node in enumerate(joint_nodes):
            for ax in ("X", "Y", "Z"):
                _set_channel(fbx, layer, node.LclRotation, ax, local_eulers[j], dt)
                _set_channel(fbx, layer, node.LclTranslation, ax, local_translations[j], dt)

        _export_scene(fbx, manager, ios, scene, save_path)

    finally:
        manager.Destroy()

    return os.path.exists(save_path)
