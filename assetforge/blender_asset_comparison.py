"""Isolated GLB views; invoked inside Blender by render_asset_comparison."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_worker import add_asset, bounds, configure_gpu

# Longest bounding-box dimension in metres: deliberate, editable display priors.
SIZES = {'ROBOT_RIDING_BIKE': 2.0, 'TREE': 1.2, 'CHAIR': 1.0,
         'TEAPOT': 0.30, 'CUP': 0.12, 'CONE': 0.70, 'MAN_HOLDING_CUP': 1.8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--setups', nargs='+', choices=['studio-agx', 'studio-standard', 'unlit-standard'],
                        default=['studio-agx', 'studio-standard', 'unlit-standard'])
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    size = SIZES.get(args.label, 1.0)
    parts = add_asset({'id': args.label, 'asset': {'kind': 'glb', 'path': str(args.asset)},
                       'world_position': [0, 0, 0], 'yaw_radians': 0}, size / 0.55)
    # Assets are isolated; imported cameras and lights are never used.
    for obj in list(bpy.data.objects):
        if obj.type in {'LIGHT', 'CAMERA'}:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    points = bounds(parts)
    target = sum(points, Vector()) / len(points)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = args.device
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 314
    if args.device == 'GPU':
        configure_gpu()
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (0.25, 0.25, 0.25, 1)
    background.inputs['Strength'].default_value = 0.25
    bpy.ops.mesh.primitive_plane_add(size=size * 200, location=(0, 0, -size * 0.008))
    floor = bpy.context.object
    material = bpy.data.materials.new('Neutral matte floor')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (0.18, 0.18, 0.18, 1)
    shader.inputs['Roughness'].default_value = 0.85
    floor.data.materials.append(material)
    for name, position, power, radius in [
        ('Key', (2, -3, 4), 180, 2.5),
        ('Fill', (-3, -1, 2), 55, 3),
        ('Rim', (1, 3, 3), 100, 2)]:
        bpy.ops.object.light_add(type='AREA', location=Vector(position) * size)
        light = bpy.context.object
        light.name = name
        light.data.energy = power * size * size
        light.data.size = radius * size
        light.rotation_euler = (target - light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = 'PERSP'
    camera.data.lens = 55
    camera.data.clip_start = size * 0.001
    camera.data.clip_end = size * 1000
    views = {'side': (1, -1.6, 0.65), 'high': (0.65, -0.8, 2.5), 'reverse': (-1, 1.6, 0.8)}
    if args.label == 'MAN_HOLDING_CUP':
        views.update({'upper-side': (1, -1.6, 0.15), 'upper-reverse': (-1, 1.6, 0.15)})
    metadata = {'label': args.label, 'longest_dimension_metres': size,
                'scale_source': 'manual display prior', 'views': {}, 'setups': {}}
    original = {m for obj in parts if obj.type == 'MESH'
                for m in obj.data.materials if m and m.use_nodes}
    # Build a true emission diagnostic from the original base-colour connection.
    # Always run emission last, since it changes the asset materials in place.
    for setup in (s for s in ('studio-agx', 'studio-standard', 'unlit-standard') if s in args.setups):
        scene.view_settings.view_transform = 'AgX' if setup == 'studio-agx' else 'Standard'
        scene.view_settings.look = 'None'
        if setup == 'unlit-standard':
            for mat in original:
                tree = mat.node_tree
                principled = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if principled is None:
                    continue
                base = principled.inputs['Base Color']
                emission = tree.nodes.new('ShaderNodeEmission')
                emission.inputs['Color'].default_value = base.default_value
                if base.is_linked:
                    tree.links.new(base.links[0].from_socket, emission.inputs['Color'])
                output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
                tree.links.new(emission.outputs[0], output.inputs['Surface'])
        metadata['setups'][setup] = {'view_transform': scene.view_settings.view_transform,
                                     'unlit_asset': setup == 'unlit-standard'}
        for name, direction in views.items():
            frame_points, frame_target = points, target
            if name.startswith('upper-'):
                minimum = Vector([min(p[i] for p in points) for i in range(3)])
                maximum = Vector([max(p[i] for p in points) for i in range(3)])
                minimum.z += (maximum.z - minimum.z) * 0.55
                frame_points = [Vector((x, y, z)) for x in (minimum.x, maximum.x)
                                for y in (minimum.y, maximum.y) for z in (minimum.z, maximum.z)]
                frame_target = (minimum + maximum) / 2
            direction = Vector(direction).normalized()
            camera.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
            # Solve camera distance from projected bounds with 20% padding.
            rotation = camera.rotation_euler.to_quaternion().inverted()
            local = [rotation @ (p - frame_target) for p in frame_points]
            tangent = math.tan(min(camera.data.angle_x, camera.data.angle_y) / 2)
            distance = max(p.z + max(abs(p.x), abs(p.y)) * 1.2 / tangent for p in local)
            camera.location = frame_target + direction * distance
            scene.render.filepath = str(args.output / f'{setup}-{name}.png')
            print('Rendering', args.label, setup, name, flush=True)
            bpy.ops.render.render(write_still=True)
            metadata['views'][name] = {'camera_location': list(camera.location), 'target': list(frame_target),
                                     'framing': 'approximate upper body; no anatomical detection' if name.startswith('upper-') else 'whole asset'}
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output / f'{setup}.blend'))
    (args.output / 'settings.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
