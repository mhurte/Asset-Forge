"""Headless orthographic renderer. Executed by Blender, never imported by the planner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def bounds(objects):
    return [obj.matrix_world @ Vector(corner) for obj in objects if obj.type == "MESH" for corner in obj.bound_box]


def configure_gpu():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            accelerators = [device for device in preferences.devices if device.type != "CPU"]
            if accelerators:
                for device in preferences.devices:
                    device.use = device.type != "CPU"
                print("Cycles GPU backend:", backend, flush=True)
                return
        except (TypeError, RuntimeError):
            continue
    raise ValueError("GPU requested but no Cycles accelerator is available.")


def add_asset(specification, tile_size):
    asset = specification["asset"]
    before = set(bpy.data.objects)
    if asset["kind"] == "glb":
        bpy.ops.import_scene.gltf(filepath=asset["path"])
    else:
        shape = asset["shape"]
        if shape == "cube":
            bpy.ops.mesh.primitive_cube_add()
        elif shape == "cone":
            bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.65, depth=1.8)
        elif shape == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.35, depth=1.8)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12)
        material = bpy.data.materials.new(f"Material_{specification['id']}")
        material.diffuse_color = asset["color"]
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = asset["color"]
        material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
        bpy.context.object.data.materials.append(material)
    imported = list(set(bpy.data.objects) - before)
    bpy.context.view_layer.update()
    points = bounds(imported)
    if not points:
        raise ValueError(f"No mesh geometry in {asset['id']}")
    minimum = Vector([min(point[axis] for point in points) for axis in range(3)])
    maximum = Vector([max(point[axis] for point in points) for axis in range(3)])
    extent = max(maximum - minimum)
    if extent <= 0:
        raise ValueError(f"Degenerate asset: {asset['id']}")
    offset = Vector([-(minimum.x + maximum.x) / 2, -(minimum.y + maximum.y) / 2, -minimum.z])
    parent = bpy.data.objects.new(specification["id"], None)
    bpy.context.collection.objects.link(parent)
    for obj in imported:
        if obj.parent not in imported:
            original_matrix = obj.matrix_world.copy()
            obj.parent = parent
            obj.matrix_world = Matrix.Translation(offset) @ original_matrix
    parent.scale = (tile_size * 0.55 / extent,) * 3
    parent.rotation_euler.z = specification["yaw_radians"]
    parent.location = specification["world_position"]
    for obj in imported:
        if obj.type == "MESH":
            obj["scene_object_id"] = specification["id"]
            obj["semantic_object_ids"] = json.dumps(specification.get("source_object_ids", [specification["id"]]))
    return imported
