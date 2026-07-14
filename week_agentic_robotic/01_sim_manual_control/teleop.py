"""Continuously control the simulated SO-101 by holding keyboard keys.

Usage:
    python 01_sim_manual_control/teleop.py

Hold these keys:
    q / a   shoulder_pan   + / -
    w / s   shoulder_lift  + / -
    e / d   elbow_flex     + / -
    r / f   wrist_flex     + / -
    t / g   wrist_roll     + / -
    y / h   gripper        open / close

    0       return all joints to home
    Esc     close the window

Mouse drag rotates or moves the camera. The scroll wheel zooms.
The upper-right picture-in-picture is the live wrist-mounted camera.
"""

import time
from pathlib import Path

import glfw
import mujoco
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "so101"
TARGET_SPEED = 0.8  # actuator target radians per second
MAX_FRAME_TIME = 0.05  # avoid a jump after pausing or dragging the window

KEY_BINDINGS = {
    glfw.KEY_Q: (0, +1), glfw.KEY_A: (0, -1),
    glfw.KEY_W: (1, +1), glfw.KEY_S: (1, -1),
    glfw.KEY_E: (2, +1), glfw.KEY_D: (2, -1),
    glfw.KEY_R: (3, +1), glfw.KEY_F: (3, -1),
    glfw.KEY_T: (4, +1), glfw.KEY_G: (4, -1),
    glfw.KEY_Y: (5, +1), glfw.KEY_H: (5, -1),
}

CONTROL_HELP = (
    "Hold Q/A  shoulder pan\n"
    "Hold W/S  shoulder lift\n"
    "Hold E/D  elbow flex\n"
    "Hold R/F  wrist flex\n"
    "Hold T/G  wrist roll\n"
    "Hold Y/H  gripper\n"
    "0 home | Esc close\n"
    "Live wrist camera: top right"
)


class KeyboardController:
    """Turn held keys into smooth, range-limited actuator targets."""

    def __init__(self, control_range, speed=TARGET_SPEED):
        self.control_range = np.asarray(control_range, dtype=float)
        self.speed = speed
        self.targets = np.zeros(len(self.control_range))
        self.held_keys = set()

    def handle_key(self, key, action):
        if key not in KEY_BINDINGS:
            return
        if action == glfw.PRESS:
            self.held_keys.add(key)
        elif action == glfw.RELEASE:
            self.held_keys.discard(key)

    def update(self, elapsed_seconds):
        direction = np.zeros(len(self.targets))
        for key in self.held_keys:
            actuator, sign = KEY_BINDINGS[key]
            direction[actuator] += sign
        self.targets += direction * self.speed * elapsed_seconds
        np.clip(
            self.targets,
            self.control_range[:, 0],
            self.control_range[:, 1],
            out=self.targets,
        )

    def home(self):
        self.held_keys.clear()
        self.targets[:] = 0.0


def run_keyboard_viewer(
    model,
    data,
    controller,
    *,
    title="SO-101 keyboard control",
    on_key_press=None,
    after_step=None,
    status_text=None,
    visible=True,
    max_frames=None,
):
    """Render MuJoCo and expose raw press/release events without shortcuts."""
    wrist_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
    )
    if wrist_camera_id == -1:
        raise RuntimeError("model is missing the required 'wrist_cam' camera")

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize a display")

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    window = glfw.create_window(1200, 800, title, None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW could not create the MuJoCo window")

    glfw.set_window_size_limits(window, 640, 480, glfw.DONT_CARE, glfw.DONT_CARE)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    camera = mujoco.MjvCamera()
    wrist_camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    wrist_scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultCamera(wrist_camera)
    mujoco.mjv_defaultOption(option)
    mujoco.mjv_defaultFreeCamera(model, camera)
    wrist_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    wrist_camera.fixedcamid = wrist_camera_id

    def key_callback(window, key, scancode, action, mods):
        del scancode, mods
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            return
        if key == glfw.KEY_0 and action == glfw.PRESS:
            controller.home()
            return

        controller.handle_key(key, action)
        if action == glfw.PRESS and on_key_press is not None:
            if on_key_press(key):
                glfw.set_window_should_close(window, True)

    def focus_callback(window, focused):
        del window
        if not focused:
            controller.held_keys.clear()

    mouse = {"x": 0.0, "y": 0.0}

    def cursor_callback(window, xpos, ypos):
        dx = xpos - mouse["x"]
        dy = ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos

        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or middle or right):
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        _, height = glfw.get_window_size(window)
        if right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, camera)

    def scroll_callback(window, xoffset, yoffset):
        del window, xoffset
        mujoco.mjv_moveCamera(
            model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            scene,
            camera,
        )

    glfw.set_key_callback(window, key_callback)
    glfw.set_window_focus_callback(window, focus_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    mujoco.mj_forward(model, data)
    last_time = time.monotonic()
    physics_time = 0.0
    frame_count = 0

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.window_should_close(window):
                break

            now = time.monotonic()
            elapsed = min(now - last_time, MAX_FRAME_TIME)
            last_time = now
            physics_time += elapsed

            while physics_time >= model.opt.timestep:
                controller.update(model.opt.timestep)
                data.ctrl[:] = controller.targets
                mujoco.mj_step(model, data)
                if after_step is not None:
                    after_step()
                physics_time -= model.opt.timestep

            width, height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0:
                glfw.wait_events_timeout(0.05)
                continue
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)

            camera_width, camera_height = model.cam_resolution[wrist_camera_id]
            camera_aspect = (
                camera_width / camera_height
                if camera_width > 0 and camera_height > 0
                else 16 / 9
            )
            inset_width = int(width * 0.32)
            inset_height = int(inset_width / camera_aspect)
            max_inset_height = int(height * 0.40)
            if inset_height > max_inset_height:
                inset_height = max_inset_height
                inset_width = int(inset_height * camera_aspect)
            margin = max(8, int(width * 0.012))
            inset = mujoco.MjrRect(
                width - inset_width - margin,
                height - inset_height - margin,
                inset_width,
                inset_height,
            )
            border = 3
            mujoco.mjr_rectangle(
                mujoco.MjrRect(
                    inset.left - border,
                    inset.bottom - border,
                    inset.width + 2 * border,
                    inset.height + 2 * border,
                ),
                0.33,
                0.78,
                1.0,
                1.0,
            )
            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                wrist_camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                wrist_scene,
            )
            mujoco.mjr_render(inset, wrist_scene, context)

            help_text = CONTROL_HELP
            if status_text is not None:
                help_text += f"\n\n{status_text()}"
            values = "\n".join(
                f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}  "
                f"{target:+.2f}"
                for i, target in enumerate(controller.targets)
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                help_text,
                values,
                context,
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                inset,
                "WRIST CAMERA",
                "",
                context,
            )
            glfw.swap_buffers(window)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_objects.xml"))
    data = mujoco.MjData(model)
    controller = KeyboardController(model.actuator_ctrlrange)
    print(__doc__)
    run_keyboard_viewer(model, data, controller)


if __name__ == "__main__":
    main()
