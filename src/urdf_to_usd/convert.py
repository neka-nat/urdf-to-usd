import os

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf, UsdUtils
from yourdfpy import URDF, Link
from scipy.spatial.transform import Rotation as R


def load_mesh_as_usd(
    stage: Usd.Stage, parent_prim: Usd.Prim, mesh_file: str
) -> Usd.Prim | None:
    if not os.path.isfile(mesh_file):
        print(f"Warning: Mesh file {mesh_file} not found.")
        return None

    mesh = trimesh.load(mesh_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Warning: {mesh_file} is not a single mesh. Skipping.")
        return None

    mesh_prim_path = parent_prim.GetPath().pathString + "/mesh"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_prim_path)

    vertices = mesh.vertices
    faces = mesh.faces

    mesh_prim.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
    face_vertex_counts = [3] * len(faces)
    mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)

    face_vertex_indices = faces.flatten().tolist()
    mesh_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)

    bbox = mesh.bounds
    extent = [Gf.Vec3f(*bbox[0]), Gf.Vec3f(*bbox[1])]
    mesh_prim.CreateExtentAttr(extent)

    return mesh_prim


def create_link_prim(
    stage: Usd.Stage, link: Link, parent_prim: Usd.Prim, urdf_base_dir: str
) -> Usd.Prim:
    link_prim_path = f"{parent_prim.GetPath().pathString}/{link.name}"
    link_prim = UsdGeom.Xform.Define(stage, link_prim_path).GetPrim()
    for visual in link.visuals:
        if visual.geometry.mesh is not None:
            mesh_prim = load_mesh_as_usd(
                stage,
                link_prim,
                os.path.join(urdf_base_dir, visual.geometry.mesh.filename),
            )
            if mesh_prim is None:
                print(f"Warning: Mesh file {visual.geometry.mesh.filename} not found.")
    return link_prim


def set_transform(prim: Usd.Prim, transform: np.ndarray) -> None:
    xyz = transform[:3, 3]
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(xyz[0], xyz[1], xyz[2]))
    rpy = np.rad2deg(R.from_matrix(transform[:3, :3]).as_euler("xyz"))
    UsdGeom.XformCommonAPI(prim).SetRotate(Gf.Vec3f(rpy[0], rpy[1], rpy[2]))


def build_usd_recurse(
    stage: Usd.Stage,
    urdf: URDF,
    link_name: str,
    parent_prim: Usd.Prim,
    urdf_base_dir: str,
) -> None:
    link_prim = create_link_prim(
        stage, urdf.link_map[link_name], parent_prim, urdf_base_dir
    )
    for joint in urdf.joint_map.values():
        if joint.parent == link_name:
            build_usd_recurse(stage, urdf, joint.child, link_prim, urdf_base_dir)
            child_link_prim = UsdGeom.Xform.Define(stage, f"{link_prim.GetPath().pathString}/{joint.child}")
            if child_link_prim is not None:
                set_transform(child_link_prim, joint.origin)


def find_root_link_name(robot: URDF) -> str:
    """childを持つが、parentを持たないlinkをroot linkとする"""
    for joint in robot.joint_map.values():
        if joint.child and joint.parent == "world":
            return joint.child
    return list(robot.link_map.values())[0].name


def convert_urdf_to_usd(
    urdf_path: str, usd_path: str, urdf_base_dir: str = "./"
) -> None:
    urdf = URDF.load(urdf_path)
    stage = Usd.Stage.CreateNew(usd_path)
    world_prim = stage.DefinePrim("/World")
    build_usd_recurse(stage, urdf, find_root_link_name(urdf), world_prim, urdf_base_dir)
    print(stage.GetRootLayer().ExportToString())
    usdz_path = usd_path.replace(".usd", ".usdz")
    UsdUtils.CreateNewUsdzPackage(usd_path, usdz_path)
