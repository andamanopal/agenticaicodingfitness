"""Run the numbered workshop lessons and report their real artifacts."""

import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from llm_providers import label_scene as label_scene_with_provider
from real_perception import capture_real_scan
from real_robot_planner import RealRobotPlanner


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCAN_DIR = ROOT / "04_active_perception" / "output"
SCENE_DIR = ROOT / "05_semantic_scene" / "output" / "latest_scene"
MANIPULATION_DIR = ROOT / "06_agentic_manipulation" / "output"
REAL_OUTPUT_DIR = ROOT / "09_real_robot_setup" / "output" / "pipeline"
REAL_SCAN_DIR = REAL_OUTPUT_DIR / "perception"
REAL_SCENE_DIR = REAL_OUTPUT_DIR / "scene"
REAL_MANIPULATION_DIR = REAL_OUTPUT_DIR / "manipulation"


SIMULATION_PATHS = {
    "mode": "simulation",
    "scan_dir": SCAN_DIR,
    "scene_dir": SCENE_DIR,
    "manipulation_dir": MANIPULATION_DIR,
}
REAL_PATHS = {
    "mode": "real",
    "scan_dir": REAL_SCAN_DIR,
    "scene_dir": REAL_SCENE_DIR,
    "manipulation_dir": REAL_MANIPULATION_DIR,
}

sys.path.insert(0, str(ROOT / "04_active_perception"))
from scan_workspace import scan_workspace  # noqa: E402

sys.path.insert(0, str(ROOT / "05_semantic_scene"))
from label_scene import merge_semantics  # noqa: E402
from reconstruct_scene import (  # noqa: E402
    GEOMETRY_CONTRACT,
    GEOMETRY_MODALITIES,
    RECONSTRUCTION_METHOD,
    SUPPORTED_SCAN_CONTRACTS,
)

sys.path.insert(0, str(ROOT / "06_agentic_manipulation"))
from robot_tools import RobotTools  # noqa: E402


class WorkflowManager:
    """Serialize long operations and retain one approved plan in memory."""

    def __init__(self, runtime, frame_hub, viewer_factory, real_runtime=None):
        self.runtime = runtime
        self.real_runtime = real_runtime
        self.frame_hub = frame_hub
        self.viewer_factory = viewer_factory
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.busy = False
        self.action = None
        self.status = "Ready"
        self.logs = ["Portal connected to the numbered workshop pipeline."]
        self.error = None
        self.tools = None
        self.real_planner = None
        self.real_plan = None
        self.plan_id = None
        self.trace = []
        self.chat_handler = None
        self.result_handler = None

    def mode(self):
        return (
            "real"
            if self.real_runtime is not None and self.real_runtime.connected
            else "simulation"
        )

    def paths(self):
        return REAL_PATHS if self.mode() == "real" else SIMULATION_PATHS

    def set_chat_handler(self, handler):
        self.chat_handler = handler

    def set_result_handler(self, handler):
        self.result_handler = handler

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def clear_memory(self, include_calibration=False):
        with self.lock:
            if self.busy:
                raise RuntimeError(f"wait for {self.action} to finish first")
            paths_for_mode = self.paths()
            scan_dir = paths_for_mode["scan_dir"]
            scene_dir = paths_for_mode["scene_dir"]
            manipulation_dir = paths_for_mode["manipulation_dir"]
            paths = [
                scan_dir / "latest_scan",
                scan_dir / ".latest_scan.in_progress",
                scene_dir,
            ]
            if include_calibration:
                paths.extend([
                    scan_dir / "empty_scan",
                    scan_dir / ".empty_scan.in_progress",
                ])
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)
            verification = manipulation_dir / "verification.png"
            verification.unlink(missing_ok=True)
            self.tools = None
            self.real_planner = None
            self.real_plan = None
            self.plan_id = None
            self.trace = []
            scope = "all scene memory" if include_calibration else "object memory"
            self.status = f"Cleared {scope}"
            self.error = None
            self.logs.append(f"✓ {self.status}")

    def invalidate_plan(self):
        self._invalidate_plan()

    def require_current_physical_plan(self, plan_id):
        with self.lock:
            plan = self.real_plan
            if plan is None or plan_id != plan.get("plan_id"):
                raise RuntimeError("the approved physical plan is missing or stale")
            if not plan.get("safe"):
                raise RuntimeError("the physical plan did not pass preflight")
            expected = dict(plan.get("planned_from_positions", {}))
        current = self.real_runtime.snapshot()["positions"]
        if set(expected) != set(current):
            raise RuntimeError("the physical plan has no valid starting joint state")
        mismatches = {
            joint: abs(float(current[joint]) - float(expected[joint]))
            for joint in expected
            if abs(float(current[joint]) - float(expected[joint]))
            > (8.0 if joint == "gripper" else 2.0)
        }
        if mismatches:
            details = ", ".join(
                (
                    f"{joint} {error:.1f}%"
                    if joint == "gripper"
                    else f"{joint} {error:.1f}°"
                )
                for joint, error in mismatches.items()
            )
            raise RuntimeError(
                "the arm moved after planning; create a new plan from the current "
                f"joint state ({details})"
            )
        return plan

    def snapshot(self):
        with self.lock:
            physical_plan = None
            if self.real_plan is not None:
                physical_plan = {
                    key: self.real_plan.get(key)
                    for key in (
                        "plan_id",
                        "object_label",
                        "destination",
                        "pick_xyz",
                        "place_xyz",
                        "planned_from_positions",
                        "safe",
                        "checks",
                        "motion_steps",
                        "not_checked",
                        "requires_physical_approval",
                    )
                }
            return {
                "busy": self.busy,
                "action": self.action,
                "status": self.status,
                "error": self.error,
                "logs": self.logs[-80:],
                "trace": self.trace,
                "plan_id": self.plan_id,
                "mode": self.mode(),
                "physical_plan": physical_plan,
            }

    def start(self, action, payload=None):
        payload = payload or {}
        with self.lock:
            if self.busy:
                raise RuntimeError(f"{self.action} is already running")
            if action in {
                "scan_empty",
                "scan_scene",
                "reconstruct_scene",
                "reconstruct",
                "make_crops",
                "label_openrouter",
                "label_manual",
                "plan",
                "execute_real",
            }:
                self.tools = None
                if action != "execute_real":
                    self.real_planner = None
                    self.real_plan = None
                self.plan_id = None
                self.trace = []
            self.busy = True
            self.action = action
            self.status = f"Starting {action.replace('_', ' ')}"
            self.error = None
            self.logs.append(f"▶ {self.status}")
        self.executor.submit(self._run, action, payload)

    def _run(self, action, payload):
        try:
            handlers = {
                "check_setup": lambda: self._command([
                    "00_getting_started/check_setup.py",
                ]),
                "kinematics": lambda: self._command([
                    "03_kinematic_pick_place/pick_place.py", "all", "--headless",
                ]),
                "scan_empty": lambda: self._scan("empty"),
                "scan_scene": lambda: self._scan("office"),
                "reconstruct_scene": self._reconstruct_scene,
                "reconstruct": lambda: self._command([
                    "05_semantic_scene/reconstruct_scene.py",
                ]),
                "make_crops": lambda: self._command([
                    "05_semantic_scene/make_object_crops.py",
                ]),
                "label_openrouter": lambda: self._command([
                    "05_semantic_scene/label_scene.py",
                ]),
                "label_manual": lambda: self._manual_labels(payload),
                "plan": lambda: self._plan(payload),
                "execute": self._execute,
                "execute_real": lambda: self._execute_real_approved(payload),
                "chat": lambda: self._chat(payload),
            }
            if action not in handlers:
                raise ValueError(f"unknown workflow action {action!r}")
            result = handlers[action]()
            with self.lock:
                self.status = f"{action.replace('_', ' ').title()} complete"
                self.logs.append(f"✓ {self.status}")
            if action == "execute_real" and self.result_handler is not None:
                self.result_handler(True, result)
        except Exception as error:
            if action == "execute_real":
                self._remove_generated_memory(include_object_scan=True)
                self._invalidate_plan()
            with self.lock:
                self.error = str(error)
                self.status = f"{action.replace('_', ' ').title()} failed"
                self.logs.append(f"✕ {error}")
            if action == "execute_real" and self.result_handler is not None:
                self.result_handler(False, str(error))
        finally:
            if self.mode() == "simulation":
                self.runtime.paused.clear()
            with self.lock:
                self.busy = False
                self.action = None

    def _chat(self, payload):
        if self.chat_handler is None:
            raise RuntimeError("chat agent is not configured")
        message = payload.get("message", "").strip()
        settings = payload.get("settings")
        if not message:
            raise ValueError("chat message is empty")
        if not settings:
            raise ValueError("LLM settings are missing")
        self.chat_handler(message, settings)

    def agent_memory(self):
        records = memory_records(self.paths())
        semantic = records["semantic"] if records["semantic_ready"] else None
        geometry = records["geometry"] if records["geometry_ready"] else None
        objects = []
        if semantic:
            objects = [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "aliases": obj.get("aliases", []),
                    "position": obj["position"],
                    "geometry_confidence": obj.get("geometry_confidence"),
                    "boundary_flags": obj.get("boundary_flags", []),
                }
                for obj in semantic.get("objects", [])
            ]
        elif geometry:
            objects = [
                {
                    "id": obj["id"],
                    "label": None,
                    "position": obj["position"],
                    "geometry_confidence": obj.get("geometry_confidence"),
                    "boundary_flags": obj.get("boundary_flags", []),
                }
                for obj in geometry.get("objects", [])
            ]

        stages = {
            "empty_table_calibration": records["background"] is not None,
            "object_scan": records["scan"] is not None,
            "geometry": records["geometry_ready"],
            "semantic_labels": records["semantic_ready"],
        }
        next_required = next(
            (name for name, ready in stages.items() if not ready),
            "ready_for_robot_commands",
        )
        return {
            "robot_mode": self.mode(),
            "stages": stages,
            "next_required": next_required,
            "scene_id": (
                records["scan"].get("scan_id") if records["scan"] else None
            ),
            "reconstruction": records["reconstruction"],
            "objects": objects,
        }

    def agent_tool(self, name, arguments, settings):
        memory = self.agent_memory()
        self._log(f"Agent tool · {name}")
        reconstruction_stages = (
            memory.get("reconstruction") or {}
        ).get("stages", {})
        reconstruction_problem = next(
            (
                (stage_name, stage)
                for stage_name, stage in reconstruction_stages.items()
                if stage.get("status") in {"review", "error"}
            ),
            None,
        )

        if name == "inspect_scene_memory":
            return memory
        if name == "capture_empty_table":
            if self.mode() == "real" and not self.real_runtime.scan_approved:
                return {
                    "status": "scan_approval_required",
                    "reason": (
                        "the physical arm must move through three taught survey poses; "
                        "approve one scan in the safety bar"
                    ),
                    **memory,
                }
            self._invalidate_plan()
            self._remove_generated_memory(include_object_scan=True)
            try:
                self._scan("empty")
            finally:
                if self.mode() == "simulation":
                    self.runtime.paused.clear()
            return {"status": "captured", **self.agent_memory()}
        if name == "scan_objects":
            if not memory["stages"]["empty_table_calibration"]:
                return {
                    "status": "blocked",
                    "reason": "empty-table calibration is missing",
                }
            if self.mode() == "real" and not self.real_runtime.scan_approved:
                return {
                    "status": "scan_approval_required",
                    "reason": (
                        "the physical arm must move through three taught survey poses; "
                        "approve one scan in the safety bar"
                    ),
                    **memory,
                }
            self._invalidate_plan()
            self._remove_generated_memory()
            try:
                self._scan("office")
            finally:
                if self.mode() == "simulation":
                    self.runtime.paused.clear()
            return {"status": "captured", **self.agent_memory()}
        if name == "reconstruct_scene":
            stages = memory["stages"]
            if not stages["empty_table_calibration"] or not stages["object_scan"]:
                return {
                    "status": "blocked",
                    "reason": "empty-table calibration and object scan are required",
                }
            self._invalidate_plan()
            self._remove_generated_memory()
            self._reconstruct_scene()
            return {"status": "reconstructed", **self.agent_memory()}
        if name == "name_objects":
            if not memory["stages"]["geometry"]:
                return {
                    "status": "blocked",
                    "reason": "scene geometry is missing",
                }
            if reconstruction_problem:
                stage_name, stage = reconstruction_problem
                return {
                    "status": "blocked",
                    "reason": (
                        f"reconstruction stage {stage_name} needs review: "
                        f"{stage.get('reason', stage.get('status'))}"
                    ),
                }
            self._invalidate_plan()
            try:
                self._name_objects(settings)
            except Exception as error:
                return {
                    "status": "manual_labels_required",
                    "reason": f"the configured model could not label images: {error}",
                    "object_ids": [obj["id"] for obj in memory["objects"]],
                }
            return {"status": "labeled", **self.agent_memory()}
        if name == "save_manual_object_labels":
            if not memory["stages"]["geometry"]:
                return {
                    "status": "blocked",
                    "reason": "scene geometry is missing",
                }
            if reconstruction_problem:
                stage_name, stage = reconstruction_problem
                return {
                    "status": "blocked",
                    "reason": (
                        f"reconstruction stage {stage_name} needs review: "
                        f"{stage.get('reason', stage.get('status'))}"
                    ),
                }
            labels = arguments.get("labels", [])
            expected_ids = [obj["id"] for obj in memory["objects"]]
            supplied_ids = [item.get("id") for item in labels]
            if len(supplied_ids) != len(set(supplied_ids)):
                return {
                    "status": "blocked",
                    "reason": "each reconstructed object must be labeled exactly once",
                    "object_ids": expected_ids,
                }
            if set(supplied_ids) != set(expected_ids):
                return {
                    "status": "blocked",
                    "reason": "provide one label for every reconstructed object ID",
                    "object_ids": expected_ids,
                }
            self._invalidate_plan()
            self._manual_labels({"labels": labels})
            return {
                "status": "labeled",
                "semantic_provider": "human",
                **self.agent_memory(),
            }
        if name == "plan_object_move":
            if not memory["stages"]["semantic_labels"]:
                return {
                    "status": "blocked",
                    "reason": "semantic object labels are missing",
                }
            if reconstruction_problem:
                stage_name, stage = reconstruction_problem
                return {
                    "status": "blocked",
                    "reason": (
                        f"reconstruction stage {stage_name} needs review: "
                        f"{stage.get('reason', stage.get('status'))}"
                    ),
                }
            return self._plan({
                "target": arguments["target"],
                "destination": arguments["destination"],
            })
        if name == "execute_planned_move":
            if self.mode() == "real":
                if self.real_plan is None:
                    return {
                        "status": "blocked",
                        "reason": "create a safe physical plan first",
                    }
                if not self.real_plan.get("safe"):
                    return {
                        "status": "blocked",
                        "reason": "the physical plan did not pass preflight",
                    }
                return {
                    "status": "physical_approval_required",
                    "reason": (
                        "physical manipulation requires approval of this exact plan "
                        "in the safety bar"
                    ),
                    "plan_id": self.real_plan["plan_id"],
                    "object_label": self.real_plan["object_label"],
                    "destination": self.real_plan["destination"],
                    "pick_xyz": self.real_plan["pick_xyz"],
                    "place_xyz": self.real_plan["place_xyz"],
                    "safe": True,
                }
            try:
                return self._execute()
            finally:
                self.runtime.paused.clear()
        raise ValueError(f"unknown agent tool {name!r}")

    def _invalidate_plan(self):
        with self.lock:
            self.tools = None
            self.real_planner = None
            self.real_plan = None
            self.plan_id = None
            self.trace = []

    def _remove_generated_memory(self, include_object_scan=False):
        paths = self.paths()
        shutil.rmtree(paths["scene_dir"], ignore_errors=True)
        if include_object_scan:
            shutil.rmtree(paths["scan_dir"] / "latest_scan", ignore_errors=True)
        (paths["manipulation_dir"] / "verification.png").unlink(missing_ok=True)

    def _command(self, arguments):
        command = [PYTHON, *arguments]
        self._log(f"$ {' '.join(arguments)}")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            self._log(line.rstrip())
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"script exited with status {return_code}")

    def _scan(self, scene):
        paths = self.paths()
        output_name = "empty_scan" if scene == "empty" else "latest_scan"
        output_dir = paths["scan_dir"] / output_name
        if self.mode() == "real":
            self._log(f"Running physical three-view {scene} scan")
            scan = capture_real_scan(self.real_runtime, output_dir, scene)
            self._log(
                f"Captured {len(scan['frames'])} calibrated physical RGB viewpoints."
            )
            return
        self.runtime.paused.set()
        self._log(f"Running Milestone 04 · {scene} wrist-camera scan")
        scan = scan_workspace(
            output_dir=output_dir,
            scene=scene,
            visualize=True,
            viewer_factory=lambda model, data: self.viewer_factory(
                model,
                data,
                "scan",
                1.0,
            ),
        )
        self._log(
            f"Captured {len(scan['frames'])} calibrated RGB viewpoints."
        )

    def _manual_labels(self, payload):
        labels = payload.get("labels", [])
        if not labels:
            raise ValueError("add at least one object label")
        geometry_path = self.paths()["scene_dir"] / "geometry.json"
        command = [
            "05_semantic_scene/label_scene.py",
            "--geometry",
            str(geometry_path),
        ]
        for item in labels:
            object_id = item.get("id", "")
            label = item.get("label", "")
            aliases_value = item.get("aliases", [])
            if not isinstance(object_id, str) or not isinstance(label, str):
                raise ValueError("object IDs and labels must be text")
            if isinstance(aliases_value, list):
                if not all(isinstance(alias, str) for alias in aliases_value):
                    raise ValueError("object aliases must be text")
                aliases = ",".join(
                    alias.strip() for alias in aliases_value if alias.strip()
                )
            elif isinstance(aliases_value, str):
                aliases = aliases_value.strip()
            else:
                raise ValueError("object aliases must be a list of text values")
            object_id = object_id.strip()
            label = label.strip()
            if not object_id or not label:
                raise ValueError("every object needs an ID and label")
            command.extend(["--label", f"{object_id}={label}"])
            if aliases:
                command.extend(["--alias", f"{object_id}={aliases}"])
        self._command(command)

    def _reconstruct_scene(self):
        paths = self.paths()
        scan_path = paths["scan_dir"] / "latest_scan" / "scan.json"
        background_path = paths["scan_dir"] / "empty_scan" / "scan.json"
        scene_dir = paths["scene_dir"]
        self._command([
            "05_semantic_scene/reconstruct_scene.py",
            "--scan",
            str(scan_path),
            "--background",
            str(background_path),
            "--output",
            str(scene_dir),
        ])
        self._command([
            "05_semantic_scene/make_object_crops.py",
            "--geometry",
            str(scene_dir / "geometry.json"),
            "--scan",
            str(scan_path),
        ])

    def _name_objects(self, settings):
        scene_dir = self.paths()["scene_dir"]
        geometry_path = scene_dir / "geometry.json"
        crops_path = scene_dir / "crops" / "crops.json"
        contact_sheet = scene_dir / "crops" / "contact_sheet.png"
        geometry = json.loads(geometry_path.read_text())
        crops = json.loads(crops_path.read_text())
        object_ids = [obj["id"] for obj in geometry["objects"]]
        crop_ids = [obj["id"] for obj in crops["objects"]]
        if crops["scene_id"] != geometry["scene_id"] or crop_ids != object_ids:
            raise RuntimeError("object crops are stale; reconstruct the scene again")

        provider = settings["provider"]
        model = settings["model"]
        self._log(f"Naming object crops with {provider} · {model}")
        labels, routed_model = label_scene_with_provider(
            settings,
            contact_sheet,
            object_ids,
        )
        semantic = merge_semantics(
            geometry,
            labels,
            f"{provider}:{routed_model}",
        )
        output_path = scene_dir / "semantic_scene.json"
        output_path.write_text(json.dumps(semantic, indent=2) + "\n")
        self._log(f"Named {len(labels)} objects with {routed_model}.")

    def _plan(self, payload):
        target = payload.get("target", "whiteboard marker").strip()
        destination = payload.get("destination", "right side")
        if self.mode() == "real":
            paths = self.paths()
            robot = self.real_runtime.snapshot()
            profile = self.real_runtime.workspace.scan_profile()
            self.real_planner = RealRobotPlanner(
                semantic_path=paths["scene_dir"] / "semantic_scene.json",
                scan_path=paths["scan_dir"] / "latest_scan" / "scan.json",
                workspace_profile=profile,
                joint_limits=robot["joint_limits"],
                current_positions=robot["positions"],
            )
            plan = self.real_planner.create_plan(target, destination)
            with self.lock:
                self.real_plan = plan
                self.plan_id = plan["plan_id"]
                self.trace = [{"tool": "physical_preflight", "result": plan}]
            self._log(
                f"Grounded '{target}' as {plan['object_label']}; "
                f"physical preflight safe={plan['safe']}."
            )
            return {
                "robot_mode": "real",
                **{
                    key: plan[key]
                    for key in (
                        "status",
                        "plan_id",
                        "object_label",
                        "destination",
                        "pick_xyz",
                        "place_xyz",
                        "safe",
                        "checks",
                        "requires_physical_approval",
                    )
                },
            }
        self.tools = RobotTools(
            execution_viewer_factory=lambda model, data: self.viewer_factory(
                model,
                data,
                "execution",
                1.0,
            )
        )
        trace = []
        scan = self.tools.scan_workspace()
        trace.append({"tool": "scan_workspace", "result": scan})
        found = self.tools.find_objects(target)
        trace.append({"tool": "find_objects", "result": found})
        if found["status"] != "found":
            raise RuntimeError("target did not resolve to exactly one visible object")
        plan = self.tools.plan_pick_and_place(
            found["matches"][0]["id"],
            destination,
        )
        trace.append({"tool": "plan_pick_and_place", "result": plan})
        preflight = self.tools.simulate_plan(plan["plan_id"])
        trace.append({"tool": "simulate_plan", "result": preflight})
        with self.lock:
            self.trace = trace
            self.plan_id = plan["plan_id"]
        self._log(
            f"Grounded '{target}' as {plan['object_label']}; "
            f"preflight safe={preflight['safe']}."
        )
        return {
            "robot_mode": "simulation",
            "status": "planned",
            "plan_id": plan["plan_id"],
            "object_label": plan["object_label"],
            "destination": destination,
            "safe": preflight["safe"],
            "checks": preflight["checks"],
        }

    def _execute(self):
        if self.tools is None or self.plan_id is None:
            raise RuntimeError("create and simulate a plan first")
        preflight = self.tools.simulations.get(self.plan_id, {})
        if not preflight.get("safe"):
            raise RuntimeError("the current plan did not pass preflight")
        self.runtime.paused.set()
        try:
            result = self.tools.execute_plan(self.plan_id, "APPROVE")
            self._append_trace("execute_plan", result)
            if result["status"] != "executed_unverified":
                raise RuntimeError(result.get("reason", "execution did not finish"))
            verification = self.tools.verify_plan(self.plan_id)
            self._append_trace("verify_plan", verification)
            if not verification["verified"]:
                raise RuntimeError("post-action visual verification did not pass")
            self._log("Approved motion completed and visual verification passed.")
            return {
                "status": "executed",
                "verified": True,
                "verification": verification,
            }
        finally:
            self.plan_id = None

    def _execute_real_approved(self, payload):
        if self.mode() != "real":
            raise RuntimeError("connect the physical SO-101 before executing a physical plan")
        plan_id = payload.get("plan_id")
        plan = self.require_current_physical_plan(plan_id)
        result = self.real_runtime.run_plan(
            plan_id,
            plan["motion_steps"],
        )
        self._append_trace("physical_execute", {
            "status": result["status"],
            "plan_id": result["plan_id"],
        })
        self._log(
            "Physical joint sequence completed. Scene memory is now stale; "
            "capture a new object scan before another plan."
        )
        self._remove_generated_memory(include_object_scan=True)
        with self.lock:
            self.real_plan = None
            self.real_planner = None
            self.plan_id = None
        return {
            "status": "executed_unverified",
            "plan_id": plan_id,
            "verified": False,
            "next_required": "rescan_and_verify",
        }

    def _append_trace(self, tool, result):
        with self.lock:
            self.trace.append({"tool": tool, "result": result})

    def _log(self, message):
        if not message:
            return
        with self.lock:
            self.logs.append(message)
            self.status = message[:160]


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_current_rgb_scan(record, mode="simulation"):
    expected_source = "real_so101" if mode == "real" else None
    return bool(
        record
        and record.get("perception_contract") in SUPPORTED_SCAN_CONTRACTS
        and record.get("sensor_modalities") == ["rgb"]
        and not any("depth" in frame for frame in record.get("frames", []))
        and (
            record.get("source") == expected_source
            if expected_source is not None
            else record.get("source") in {None, "simulation"}
        )
    )


def memory_records(paths=None):
    paths = paths or SIMULATION_PATHS
    scan_dir = paths["scan_dir"]
    scene_dir = paths["scene_dir"]
    mode = paths["mode"]
    background = read_json(scan_dir / "empty_scan" / "scan.json")
    scan = read_json(scan_dir / "latest_scan" / "scan.json")
    background = background if is_current_rgb_scan(background, mode) else None
    scan = scan if is_current_rgb_scan(scan, mode) else None
    if background and scan and background.get("camera") != scan.get("camera"):
        scan = None
    if (
        background
        and scan
        and background.get("workspace_profile_sha256")
        != scan.get("workspace_profile_sha256")
    ):
        scan = None
    geometry = read_json(scene_dir / "geometry.json")
    crops = read_json(scene_dir / "crops" / "crops.json")
    semantic = read_json(scene_dir / "semantic_scene.json")
    reconstruction = read_json(scene_dir / "depth" / "diagnostics.json")
    if reconstruction and (
        not scan
        or reconstruction.get("scene_id") != scan.get("scan_id")
        or not background
        or reconstruction.get("background_scan_id") != background.get("scan_id")
    ):
        reconstruction = None
    geometry_ready = bool(
        background
        and scan
        and geometry
        and crops
        and geometry.get("scene_id") == scan.get("scan_id")
        and geometry.get("background_scan_id") == background.get("scan_id")
        and geometry.get("perception_contract") == GEOMETRY_CONTRACT
        and geometry.get("sensor_modalities") == GEOMETRY_MODALITIES
        and geometry.get("uses_privileged_simulator_data") is False
        and geometry.get("method") == RECONSTRUCTION_METHOD
        and crops.get("scene_id") == geometry.get("scene_id")
    )
    semantic_ready = bool(
        geometry_ready
        and semantic
        and semantic.get("scene_id") == geometry.get("scene_id")
        and semantic.get("background_scan_id")
        == geometry.get("background_scan_id")
        and semantic.get("perception_contract") == GEOMETRY_CONTRACT
        and semantic.get("sensor_modalities") == GEOMETRY_MODALITIES
        and semantic.get("uses_privileged_simulator_data") is False
        and semantic.get("geometry_method") == RECONSTRUCTION_METHOD
    )
    return {
        "background": background,
        "scan": scan,
        "geometry": geometry,
        "crops": crops,
        "semantic": semantic,
        "reconstruction": reconstruction,
        "geometry_ready": geometry_ready,
        "semantic_ready": semantic_ready,
    }


def artifact_state(paths=None):
    paths_for_mode = paths or SIMULATION_PATHS
    scan_dir = paths_for_mode["scan_dir"]
    scene_dir = paths_for_mode["scene_dir"]
    manipulation_dir = paths_for_mode["manipulation_dir"]
    records = memory_records(paths_for_mode)
    paths = {}
    artifacts = []

    def add(artifact_id, title, path, live=False):
        path = Path(path)
        if not path.exists():
            return
        paths[artifact_id] = path
        artifacts.append({
            "id": artifact_id,
            "title": title,
            "url": f"/api/artifact/{artifact_id}?v={path.stat().st_mtime_ns}",
            "updated": datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            "live": live,
        })

    for output_name, label in (
        ("empty_scan", "Calibration"),
        ("latest_scan", "Object scan"),
    ):
        staging = scan_dir / f".{output_name}.in_progress"
        for viewpoint in ("survey_left", "survey_center", "survey_right"):
            add(
                f"live_{output_name}_{viewpoint}",
                f"Live {label} · {viewpoint.removeprefix('survey_').title()}",
                staging / f"{viewpoint}.png",
                live=True,
            )

    if records["background"]:
        add(
            "calibration_scan",
            "Empty-table calibration",
            scan_dir / "empty_scan" / "contact_sheet.png",
        )
    if records["scan"]:
        add(
            "rgb_scan",
            "RGB object scan",
            scan_dir / "latest_scan" / "contact_sheet.png",
        )
    add(
        "background_alignment",
        "Empty-table pose alignment",
        scene_dir / "masks" / "background_alignment_contact_sheet.png",
    )
    add(
        "foreground_masks",
        "Foreground object masks",
        scene_dir / "masks" / "foreground_contact_sheet.png",
    )
    add(
        "raw_depth",
        "Raw learned depth",
        scene_dir / "depth" / "raw_depth_contact_sheet.png",
    )
    add(
        "aligned_depth",
        "Table-aligned metric depth",
        scene_dir / "depth" / "aligned_depth_contact_sheet.png",
    )
    add(
        "depth_support",
        "Cross-view depth agreement",
        scene_dir / "depth" / "support_contact_sheet.png",
    )
    if records["geometry_ready"]:
        add(
            "height_map",
            "Fused metric height map",
            scene_dir / "height_map.png",
        )
        add(
            "top_view",
            "Fused cloud + planner fit",
            scene_dir / "topdown.png",
        )
        add(
            "object_crops",
            "Semantic object crops",
            scene_dir / "crops" / "contact_sheet.png",
        )
    if records["semantic_ready"]:
        add(
            "verification",
            "Latest action verification",
            manipulation_dir / "verification.png",
        )
    return artifacts, paths


def scene_state(paths=None):
    records = memory_records(paths or SIMULATION_PATHS)
    geometry = records["geometry"] if records["geometry_ready"] else None
    semantic = records["semantic"] if records["semantic_ready"] else None
    if geometry is not None:
        geometry = {
            key: value
            for key, value in geometry.items()
            if key not in {"points", "occupied_voxels_xyz"}
        }
    return {
        "geometry": geometry,
        "semantic": semantic,
        "reconstruction": records["reconstruction"],
    }


def readiness_state(paths=None):
    records = memory_records(paths or SIMULATION_PATHS)
    background_ready = records["background"] is not None
    scan_ready = records["scan"] is not None
    geometry_ready = records["geometry_ready"]
    semantic_ready = records["semantic_ready"]
    steps = [
        ("00", "Environment", True),
        ("01", "Manual control", True),
        ("04A", "Empty calibration", background_ready),
        ("04B", "Occupied scan", scan_ready),
        ("05", "Scene geometry", geometry_ready),
        ("05S", "Semantic labels", semantic_ready),
        ("06", "Agentic motion", semantic_ready),
    ]
    return [
        {"number": number, "label": label, "ready": ready}
        for number, label, ready in steps
    ]
