"""Render ImageDream-compatible multi-view data from an articulated .blend.

Given a `.blend` produced by `make_articulated_blend.py`, this renders, for each
of N fixed views around the object:

  * the articulation **video frames** (textured, all animation frames),
  * the rest-state **image** (frame 1, textured), and
  * the rest-state **binary mask** of the articulated part (exactly aligned to
    the image),

and writes the camera **pose** per view.

Camera / scene from the blend
-----------------------------
The starting camera pose and lens are read from the blend's active camera:
position that camera so the object fits the frame (adjust its distance / height
/ lens), then the N views orbit the origin at that **radius** and **elevation**,
tracking the origin. View ``i`` is at azimuth ``start_azimuth + i * 360/N``,
where ``start_azimuth`` is the placed camera's azimuth (override any of these
with ``--dist`` / ``--elev`` / ``--azim-start`` / ``--lens`` / ``--sensor``).

Lighting, world, resolution and engine are taken from the blend as well; supply
``--lighting-blend`` / ``--resolution`` / ``--engine`` / ``--samples`` only to
override them. This matches the project's dataloader convention: the pose file is
one line ``azimuth elevation distance`` (spherical, azimuth in [-180,180]).

The binary mask is rendered via Cycles' Object-Index pass (the articulated part
carries ``pass_index = 1``) + an ID-Mask compositor node, so it is
occlusion-correct and pixel-aligned with the textured image.

Usage (with the bpy-enabled `atop` env, or inside Blender)::

    python src/dataset/render_views.py -- \
        --blend datasetv0/StorageFurniture/35059/blends/35059_0_hinge.blend
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys

import bpy


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, distance: float):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    )


def _look_at_center(scene, cam):
    """Return the world point the camera aims at: its TRACK_TO target, else a
    'look_at' empty, else the origin."""
    for constraint in cam.constraints:
        if constraint.type == "TRACK_TO" and constraint.target is not None:
            return constraint.target.matrix_world.to_translation()
    empty = next((o for o in scene.objects
                  if o.type == "EMPTY" and o.name.startswith(("look_at", "cam_target"))), None)
    if empty is not None:
        return empty.matrix_world.to_translation()
    return (0.0, 0.0, 0.0)


def read_camera_from_blend(scene):
    """Read the starting camera pose from the blend's active camera and its
    look-at target.

    Position the camera (and move the ``look_at`` empty) in the blend so the
    object fits the frame; the N views then orbit the look-at point at this
    radius and elevation. Returns the look-at ``center``, radius, elevation &
    azimuth (deg) relative to it, and the lens / sensor. None if no camera.
    """
    cam = scene.camera or next((o for o in scene.objects if o.type == "CAMERA"), None)
    if cam is None:
        return None
    center = _look_at_center(scene, cam)
    loc = cam.matrix_world.to_translation()
    rel = (loc[0] - center[0], loc[1] - center[1], loc[2] - center[2])
    radius = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
    elevation = math.degrees(math.asin(rel[2] / radius)) if radius > 0 else 0.0
    azimuth = math.degrees(math.atan2(rel[1], rel[0]))
    return {"cam": cam, "center": center, "radius": radius, "elevation": elevation,
            "azimuth": azimuth, "lens": cam.data.lens, "sensor": cam.data.sensor_width}


def aim_camera_at_target(scene, cam, center, lens_mm: float, sensor_mm: float):
    """Ensure `cam` tracks a target placed at `center`, and set its lens, so it
    keeps looking at that point as it orbits. Reuses an existing TRACK_TO target."""
    cam.data.lens_unit = "MILLIMETERS"
    cam.data.lens = lens_mm
    cam.data.sensor_width = sensor_mm

    target = None
    for constraint in cam.constraints:
        if constraint.type == "TRACK_TO" and constraint.target is not None:
            target = constraint.target
            break
    if target is None:
        target = bpy.data.objects.new("cam_target", None)
        scene.collection.objects.link(target)
        track = cam.constraints.new(type="TRACK_TO")
        track.track_axis = "TRACK_NEGATIVE_Z"
        track.up_axis = "UP_Y"
        track.target = target
    target.location = (center[0], center[1], center[2])

    scene.camera = cam
    return cam


def place_camera(cam, center, azimuth_deg: float, elevation_deg: float, distance: float):
    """Place the camera on the sphere of `distance` around `center`."""
    off = spherical_to_cartesian(azimuth_deg, elevation_deg, distance)
    cam.location = (center[0] + off[0], center[1] + off[1], center[2] + off[2])
    bpy.context.view_layer.update()


def pose_line(cam, center) -> str:
    """Return 'azimuth elevation distance' (spherical) of the camera relative to
    the look-at `center`, azimuth in [-180, 180] -- the format the dataloader
    expects."""
    loc = cam.matrix_world.to_translation()
    rel = (loc[0] - center[0], loc[1] - center[1], loc[2] - center[2])
    radius = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
    azimuth = math.degrees(math.atan2(rel[1], rel[0]))
    elevation = math.degrees(math.asin(rel[2] / radius)) if radius > 0 else 0.0
    return f"{azimuth} {elevation} {radius}\n"


# --------------------------------------------------------------------------- #
# Scene look
# --------------------------------------------------------------------------- #
def apply_lighting_from_blend(scene, blend_path: str) -> None:
    """Replace the scene lighting with the lights (and world) authored in an
    external .blend. Only LIGHT objects and the World datablock are pulled in;
    any meshes/cameras in that file are ignored. Author the lights for an object
    centred at the origin, Z-up (same world as the rendered object)."""
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)

    with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
        dst.objects = list(src.objects)
        dst.worlds = list(src.worlds)

    n_lights = 0
    for obj in dst.objects:
        if obj is None:
            continue
        if obj.type == "LIGHT":
            scene.collection.objects.link(obj)
            n_lights += 1
        else:
            bpy.data.objects.remove(obj, do_unlink=True)

    if dst.worlds:
        scene.world = dst.worlds[0]
    print(f"[render_views] lighting from {os.path.basename(blend_path)}: "
          f"{n_lights} light(s), world={'imported' if dst.worlds else 'kept existing'}")


# --------------------------------------------------------------------------- #
# Render engine / passes
# --------------------------------------------------------------------------- #
def enable_gpu(scene) -> str:
    """Enable Cycles GPU (OptiX/CUDA) if available; return the device used."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = (d.type == backend)
                scene.cycles.device = "GPU"
                return backend
        except Exception:
            continue
    scene.cycles.device = "CPU"
    return "CPU"


def set_mask_mode(scene) -> None:
    """Binary mask render: Cycles Object-Index pass -> ID Mask -> Composite.

    The articulated part has pass_index == 1; the ID-Mask node outputs a 0/1
    matte of only the (front-most) visible pixels of that part.
    """
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0  # no dithering -> strictly 0/255 mask
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"

    view_layer = scene.view_layers[0]
    view_layer.use_pass_object_index = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rlayers = tree.nodes.new("CompositorNodeRLayers")
    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    id_mask.use_antialiasing = False  # hard 0/1 -> strictly binary mask
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rlayers.outputs["IndexOB"], id_mask.inputs["ID value"])
    tree.links.new(id_mask.outputs["Alpha"], composite.inputs["Image"])


def render_to(scene, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def _find_ffmpeg():
    """Locate an ffmpeg CLI (PATH, then next to this interpreter / conda base)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    bindir = os.path.dirname(sys.executable)
    for cand in (os.path.join(bindir, "ffmpeg"),
                 os.path.join(bindir, "..", "..", "..", "bin", "ffmpeg")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def encode_mp4(frames_dir: str, out_mp4: str, fps: int, start_number: int) -> None:
    """Encode frame_###.png in `frames_dir` into an H.264 mp4 (keeps the frames)."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("[render_views] ffmpeg not found -- skipping mp4 (frames kept)")
        return
    # mpeg4 is available in every ffmpeg build (libx264 often is not).
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps), "-start_number", str(start_number),
        "-i", os.path.join(frames_dir, "frame_%03d.png"),
        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", out_mp4,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, help="path to an articulated .blend")
    p.add_argument("--out-dir", default=None,
                   help="output root (default: <model_dir>/render/<blend_stem>)")
    p.add_argument("--lighting-blend", default=None,
                   help="optional .blend whose lights/world replace the blend's own lighting")
    p.add_argument("--nviews", type=int, default=8)
    # The following default to values read from the blend's camera/scene; pass
    # any of them to override just that one.
    p.add_argument("--azim-start", type=float, default=None,
                   help="azimuth (deg) of the first view (default: blend camera). "
                        "The N views are azim_start + i*360/N")
    p.add_argument("--elev", type=float, default=None,
                   help="camera elevation deg (default: from blend camera)")
    p.add_argument("--dist", type=float, default=None,
                   help="orbit radius (default: from blend camera)")
    p.add_argument("--lens", type=float, default=None,
                   help="camera lens mm (default: from blend camera)")
    p.add_argument("--sensor", type=float, default=None,
                   help="camera sensor width mm (default: from blend camera)")
    p.add_argument("--resolution", type=int, default=None,
                   help="square image size px (default: from blend)")
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE_NEXT"], default=None,
                   help="engine for the textured renders (default: from blend); masks use Cycles")
    p.add_argument("--samples", type=int, default=None,
                   help="Cycles samples for beauty (default: from blend)")
    p.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    p.add_argument("--fps", type=int, default=None,
                   help="mp4 frame rate (default: the blend's scene fps)")
    p.add_argument("--no-mp4", action="store_true",
                   help="do not encode per-view mp4 videos (only keep PNG frames)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=args.blend)
    scene = bpy.context.scene

    stem = os.path.splitext(os.path.basename(args.blend))[0]
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.blend)))  # .../<id>
    out_root = args.out_dir or os.path.join(model_dir, "render", stem)

    # Everything about the look (lighting, world, white-background compositor,
    # film, view transform, resolution, engine, samples) is taken from the blend.
    # The CLI values below only override when explicitly given.
    if args.resolution is not None:
        scene.render.resolution_x = scene.render.resolution_y = args.resolution
    if args.engine:
        scene.render.engine = args.engine
    if args.samples is not None and scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
    if args.lighting_blend:
        apply_lighting_from_blend(scene, args.lighting_blend)  # else: use blend's own
    if args.device == "GPU":
        print(f"[render_views] Cycles device: {enable_gpu(scene)}")

    # Camera pose comes from the blend (adjust radius/height there to fit the FOV);
    # any of these can be overridden on the CLI.
    info = read_camera_from_blend(scene)
    if info is None:
        raise SystemExit("[render_views] no camera in the blend -- add/position one first.")
    center = info["center"]
    radius = args.dist if args.dist is not None else info["radius"]
    elevation = args.elev if args.elev is not None else info["elevation"]
    start_azimuth = args.azim_start if args.azim_start is not None else info["azimuth"]
    lens = args.lens if args.lens is not None else info["lens"]
    sensor = args.sensor if args.sensor is not None else info["sensor"]
    cam = aim_camera_at_target(scene, info["cam"], center, lens, sensor)
    print(f"[render_views] look-at={tuple(round(c, 3) for c in center)} "
          f"radius={radius:.3f} elev={elevation:.1f} start_azim={start_azimuth:.1f} "
          f"lens={lens:.1f}mm sensor={sensor:.1f}mm | {args.nviews} views")

    frame_start, frame_end = scene.frame_start, scene.frame_end
    step = 360.0 / args.nviews
    azimuths = [start_azimuth + i * step for i in range(args.nviews)]

    # --- 1) textured beauty: rest image (frame 1) + articulation video ------
    # Use the blend's own render settings as-is; only ensure we render the
    # textured object (not the debug orange/blue highlight).
    scene["mask_mode"] = 0
    for vi, azim in enumerate(azimuths):
        place_camera(cam, center, azim, elevation, radius)
        tag = f"{vi:02d}"

        # camera pose (spherical, relative to look-at) -- what the model consumes
        pose_path = os.path.join(out_root, "poses", f"{tag}.txt")
        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        with open(pose_path, "w") as f:
            f.write(pose_line(cam, center))

        # rest-state image = first frame
        scene.frame_set(frame_start)
        render_to(scene, os.path.join(out_root, "images", f"{tag}.png"))

        # articulation video frames (clear stale frames first so mp4 is exact)
        frames_dir = os.path.join(out_root, "video", tag)
        shutil.rmtree(frames_dir, ignore_errors=True)
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            render_to(scene, os.path.join(frames_dir, f"frame_{frame:03d}.png"))

        # encode those frames into an mp4 (frames are kept alongside)
        if not args.no_mp4:
            fps = args.fps if args.fps is not None else scene.render.fps
            encode_mp4(frames_dir, os.path.join(out_root, "video", f"{tag}.mp4"),
                       fps, frame_start)
        print(f"[render_views] view {tag}: azim={azim:.0f} beauty done")

    # --- 2) binary mask of the articulated part at rest (frame 1) -----------
    set_mask_mode(scene)
    scene.frame_set(frame_start)
    for vi, azim in enumerate(azimuths):
        place_camera(cam, center, azim, elevation, radius)
        render_to(scene, os.path.join(out_root, "masks", f"{vi:02d}.png"))
    print(f"[render_views] masks done -> {out_root}")


if __name__ == "__main__":
    main()
