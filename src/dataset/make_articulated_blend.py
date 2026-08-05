"""Create animated, textured Blender (.blend) files for articulated SAPIEN shapes.

For a PartNet-Mobility (SAPIEN) shape, this loads the textured part meshes and,
for each movable part, builds a Blender scene where that part is rigged to a
pivot empty and key-framed across its joint range (hinge rotation or slider
translation). The result is a `.blend` you can open and press *play* to watch the
part articulate, with the original textures applied.

One `.blend` is written per movable part:

    <out_dir>/<model_id>_<part_idx>_<joint>.blend

Input (per model), as found under `datasetv0/<Category>/<model_id>/`:
    result.json          part hierarchy (leaf -> .obj file stems)
    mobility_v2.json      per-part joint annotations (axis + limits)
    textured_objs/*.obj    per-part textured meshes (+ .mtl)
    images/*.jpg           textures referenced by the .mtl files

Coordinate handling
-------------------
SAPIEN meshes are Y-up. Blender's default OBJ importer rotates them into its
Z-up world via (x, y, z) -> (x, -z, y). The joint axis (origin + direction) in
`mobility_v2.json` is in the raw Y-up frame, so we apply the same mapping to it;
the rigged animation then matches the upright, textured geometry exactly.

Usage (with a bpy-enabled interpreter, or inside Blender)::

    python src/dataset/make_articulated_blend.py \
        --data-root datasetv0 --category Refrigerator --model-id 10036

    blender --background --python src/dataset/make_articulated_blend.py -- \
        --data-root datasetv0 --category Refrigerator --model-id 10036
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import bpy
from mathutils import Vector


# --------------------------------------------------------------------------- #
# SAPIEN parsing
# --------------------------------------------------------------------------- #
def parse_parts_data(part: dict, id_to_objs: Dict[int, dict]) -> List[str]:
    """Recursively flatten result.json, mapping each part id to its .obj stems."""
    part_id = part["id"]
    id_to_objs[part_id] = {"name": part["name"]}
    if "children" in part:
        objs: List[str] = []
        for child in part["children"]:
            objs += parse_parts_data(child, id_to_objs)
    else:
        objs = list(part["objs"])
    id_to_objs[part_id]["objs"] = objs
    return objs


def movable_parts(art_data: list, id_to_objs: Dict[int, dict]) -> List[dict]:
    """Return the hinge/slider parts, each as a dict with its obj stems + joint."""
    parts = []
    for entry in art_data:
        joint = entry.get("joint")
        if joint not in ("hinge", "slider") or not entry.get("jointData"):
            continue
        part_ids = [p["id"] for p in entry["parts"]]
        stems: List[str] = []
        for pid in part_ids:
            stems += id_to_objs[pid]["objs"]
        parts.append({"joint": joint, "jointData": entry["jointData"],
                      "name": entry.get("name", ""), "stems": stems})
    return parts


def joint_range(joint_data: dict, joint_type: str) -> Tuple[float, float]:
    """Return the (start, end) joint value; degrees for hinge, distance for slider."""
    limit = joint_data["limit"]
    if limit.get("noLimit", False):
        return (0.0, 360.0) if joint_type == "hinge" else (0.0, 1.0)
    return float(limit["a"]), float(limit["b"])


# SAPIEN (Y-up) -> Blender (Z-up), matching Blender's default OBJ import.
def to_blender(v) -> Vector:
    return Vector((v[0], -v[2], v[1]))


# --------------------------------------------------------------------------- #
# Blender scene building
# --------------------------------------------------------------------------- #
def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_world_and_lights(scene) -> None:
    """Add a World, lights and a camera.

    `read_factory_settings(use_empty=True)` yields a scene with NO world
    datablock; switching to rendered shading then dereferences a null world and
    crashes EEVEE. Creating a world (plus lights) fixes the crash and makes the
    rendered view actually lit.
    """
    # World: soft, even white fill so every side of the object stays visible.
    # Kept low so the sun is the key light; no directional shadows come from it.
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.5
    scene.world = world

    # Key light: a single shadowless sun, angled from above-front so surface
    # features read through shading without casting any shadows.
    sun_data = bpy.data.lights.new(name="sun", type="SUN")
    sun_data.energy = 4.0        # user-requested power in [3, 5]
    sun_data.use_shadow = False  # no shadow effects
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(30))
    scene.collection.objects.link(sun)

    # White render background: render the lit object with a transparent film and
    # composite it over solid white.
    scene.render.film_transparent = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rlayers = tree.nodes.new("CompositorNodeRLayers")
    over = tree.nodes.new("CompositorNodeAlphaOver")
    over.inputs[1].default_value = (1.0, 1.0, 1.0, 1.0)  # white background
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rlayers.outputs["Image"], over.inputs[2])
    tree.links.new(over.outputs["Image"], composite.inputs["Image"])

    # Look-at target: the point the camera aims at and the render views orbit.
    # Move this empty in the blend to re-centre framing (render_views.py reads it).
    look_at = bpy.data.objects.new("look_at", None)
    look_at.empty_display_type = "PLAIN_AXES"
    look_at.empty_display_size = 0.3
    look_at.location = (0.0, 0.0, 0.0)
    scene.collection.objects.link(look_at)

    # A camera that tracks the look-at target. Move/zoom this camera in the blend
    # to fit the object in frame; render_views.py reads its pose, lens and target.
    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    cam.location = (3.0, -3.0, 2.0)
    scene.collection.objects.link(cam)
    track = cam.constraints.new(type="TRACK_TO")
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    track.target = look_at
    scene.camera = cam

    # Render resolution baked into the blend (render_views.py reads it).
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024


def import_parts(textured_objs_dir: str, stems: List[str]) -> Dict[str, List]:
    """Import each part .obj (upright, with textures); return stem -> [objects]."""
    stem_to_objects: Dict[str, List] = {}
    for stem in stems:
        path = os.path.join(textured_objs_dir, f"{stem}.obj")
        if not os.path.exists(path):
            print(f"  [warn] missing obj: {path}")
            continue
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=path)  # default axes -> upright + textured
        new_objects = [o for o in bpy.data.objects if o not in before]
        # SAPIEN meshes contain invalid geometry (degenerate faces / bad normals)
        # that crashes EEVEE's GPU batch build on macOS/Metal in rendered mode.
        # validate() repairs the mesh data in place.
        for obj in new_objects:
            if obj.type == "MESH":
                obj.data.validate(verbose=False)
                obj.data.update()
        stem_to_objects[stem] = new_objects
    return stem_to_objects


def deduplicate_materials() -> int:
    """Merge materials that share a base name (e.g. 'mat', 'mat.001', ...).

    Importing each part as a separate OBJ produces one copy of every material
    per file, so a shape can end up with hundreds of near-identical materials
    (`material_0_0`, `material_0_0.001`, ...). Each becomes its own GPU shader;
    compiling that many at once crashes EEVEE on macOS/Metal when entering
    rendered mode. Collapsing them to one material per base name keeps the shader
    count tiny. Returns the number of materials removed.
    """
    canonical: Dict[str, "bpy.types.Material"] = {}
    for mat in list(bpy.data.materials):
        canonical.setdefault(mat.name.split(".")[0], mat)

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if slot.material is not None:
                slot.material = canonical[slot.material.name.split(".")[0]]

    removed = 0
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# Part segmentation labels / masks
# --------------------------------------------------------------------------- #
# Two-tone highlight colours (sRGB): the articulated part is orange and
# everything else (base + other parts) is blue -- the highlight style used in
# shape-analysis result figures. The mask render is *shaded* (diffuse), so the
# parts keep their form, not a flat fill.
BASE_COLOR = (44, 169, 214)      # azure blue  (static / base)
MOVABLE_COLOR = (242, 133, 32)   # orange      (articulated part)


def _srgb_to_linear(c: float) -> float:
    """Convert an sRGB channel [0,1] to linear so object.color renders as the
    intended sRGB colour through Blender's Standard (sRGB) view transform."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_rgba(rgb255) -> tuple:
    r, g, b = rgb255
    return (_srgb_to_linear(r / 255.0), _srgb_to_linear(g / 255.0),
            _srgb_to_linear(b / 255.0), 1.0)


def color_objects(mesh_objects: List, target_objects: List) -> None:
    """Colour the articulated part orange and everything else blue.

    The colour is stored on `object.color` (read by the mask material via an
    Object-Info node), together with a binary `pass_index` / `part_label`
    (1 = articulated part, 0 = base)."""
    target = set(target_objects)
    for obj in mesh_objects:
        movable = obj in target
        obj.color = _linear_rgba(MOVABLE_COLOR if movable else BASE_COLOR)
        obj.pass_index = 1 if movable else 0
        obj["part_label"] = obj.pass_index


def _inject_mask_switch(material, scene) -> None:
    """Insert a Mix Shader that flips a material between its texture and a
    *shaded* diffuse fill of the object's highlight colour (orange/blue),
    driven by scene['mask_mode']."""
    nt = material.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
              next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None))
    if out is None or not out.inputs["Surface"].is_linked:
        return
    beauty = out.inputs["Surface"].links[0].from_socket

    obj_info = nt.nodes.new("ShaderNodeObjectInfo")
    shaded = nt.nodes.new("ShaderNodeBsdfDiffuse")  # lit -> keeps the shaded look
    nt.links.new(obj_info.outputs["Color"], shaded.inputs["Color"])

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(beauty, mix.inputs[1])          # Fac 0 -> textured beauty
    nt.links.new(shaded.outputs[0], mix.inputs[2])  # Fac 1 -> shaded highlight
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    fcurve = mix.inputs["Fac"].driver_add("default_value")
    driver = fcurve.driver
    driver.type = "SCRIPTED"
    driver.expression = "mask_mode"
    var = driver.variables.new()
    var.name = "mask_mode"
    var.type = "SINGLE_PROP"
    var.targets[0].id_type = "SCENE"
    var.targets[0].id = scene
    var.targets[0].data_path = '["mask_mode"]'


def add_mask_mode(scene) -> None:
    """Enable a scene-wide highlight toggle: set scene['mask_mode'] = 1 to render
    the shaded orange/blue highlight, 0 for the textured beauty render."""
    scene["mask_mode"] = 0
    scene.render.film_transparent = True  # clean (transparent) background
    # Standard view transform so the highlight colours render true (AgX/Filmic
    # would tonemap and shift them).
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_layers[0].use_pass_object_index = True  # bonus IndexOB pass
    except Exception:
        pass
    for material in list(bpy.data.materials):
        if material.use_nodes:
            _inject_mask_switch(material, scene)


def parent_keep_transform(child, parent) -> None:
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def set_linear(obj) -> None:
    """Force linear interpolation so the part articulates at a constant rate."""
    action = obj.animation_data.action if obj.animation_data else None
    if not action:
        return
    for fcurve in action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "LINEAR"


def animate_part(target_objects: List, joint: str, joint_data: dict, n_frames: int):
    """Rig `target_objects` to a pivot empty and key-frame the joint over n_frames."""
    origin = to_blender(joint_data["axis"]["origin"])
    direction = to_blender(joint_data["axis"]["direction"])
    a, b = joint_range(joint_data, joint)

    pivot = bpy.data.objects.new("joint_pivot", None)
    pivot.empty_display_type = "ARROWS"
    pivot.empty_display_size = 0.2
    pivot.location = origin
    pivot.rotation_mode = "AXIS_ANGLE"
    pivot.rotation_axis_angle = (0.0, direction.x, direction.y, direction.z)
    bpy.context.scene.collection.objects.link(pivot)
    bpy.context.view_layer.update()

    for obj in target_objects:
        parent_keep_transform(obj, pivot)

    for i in range(n_frames):
        frame = i + 1
        value = a + (b - a) * (i / (n_frames - 1)) if n_frames > 1 else a
        if joint == "hinge":
            pivot.rotation_axis_angle = (math.radians(value),
                                         direction.x, direction.y, direction.z)
            pivot.keyframe_insert("rotation_axis_angle", frame=frame)
        else:  # slider: translate along the (raw-magnitude) direction
            pivot.location = origin + direction * value
            pivot.keyframe_insert("location", frame=frame)

    set_linear(pivot)
    return pivot


def build_blend(model_dir: str, part: dict, all_stems: List[str],
                n_frames: int, out_path: str) -> None:
    """Build a scene with the full model, animate `part`, and save `out_path`."""
    reset_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = n_frames

    setup_world_and_lights(scene)
    stem_to_objects = import_parts(os.path.join(model_dir, "textured_objs"), all_stems)
    deduplicate_materials()  # collapse duplicate shaders (macOS/Metal crash fix)

    target_objects = [o for stem in part["stems"]
                      for o in stem_to_objects.get(stem, [])]
    if not target_objects:
        print(f"  [warn] no imported geometry for movable part; skipping {out_path}")
        return

    # Colour the articulated part orange / base blue, and add the render toggle.
    mesh_objects = [o for objs in stem_to_objects.values() for o in objs
                    if o.type == "MESH"]
    color_objects(mesh_objects, target_objects)
    add_mask_mode(scene)

    animate_part(target_objects, part["joint"], part["jointData"], n_frames)

    scene.frame_set(1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Pack textures into the .blend so the single file is self-contained and
    # renders correctly when copied to another machine (e.g. viewing on a Mac).
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"  saved {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="dataset root (e.g. datasetv0)")
    parser.add_argument("--category", required=True, help="category (e.g. Refrigerator)")
    parser.add_argument("--model-id", required=True, help="shape id (e.g. 10036)")
    parser.add_argument("--frames", type=int, default=30, help="animation length in frames")
    parser.add_argument("--out-dir", default=None,
                        help="output dir for .blend files (default: <model_dir>/blends)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    model_dir = os.path.join(args.data_root, args.category, args.model_id)
    out_dir = args.out_dir or os.path.join(model_dir, "blends")

    with open(os.path.join(model_dir, "result.json")) as f:
        parts_data = json.load(f)[0]
    with open(os.path.join(model_dir, "mobility_v2.json")) as f:
        art_data = json.load(f)

    id_to_objs: Dict[int, dict] = {}
    parse_parts_data(parts_data, id_to_objs)
    all_stems = id_to_objs[parts_data["id"]]["objs"]

    parts = movable_parts(art_data, id_to_objs)
    if not parts:
        print(f"[{args.model_id}] no hinge/slider parts found; nothing to do.")
        return

    print(f"[{args.model_id}] {len(parts)} movable part(s), {len(all_stems)} meshes")
    for idx, part in enumerate(parts):
        out_path = os.path.join(out_dir, f"{args.model_id}_{idx}_{part['joint']}.blend")
        print(f"- part {idx}: {part['name']} ({part['joint']})")
        build_blend(model_dir, part, all_stems, args.frames, out_path)


if __name__ == "__main__":
    main()
