"""Local workshop portal for the MuJoCo-first SO-101 lessons."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat_agent import ChatAgent
from hardware_setup import HardwareSetupManager
from real_robot_runtime import RealRobotRuntime
from robot_runtime import FrameHub, PortalViewer, RobotRuntime
from workflow import WorkflowManager, artifact_state, readiness_state, scene_state


HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
GUIDE_PATH = HERE.parent / "workshop-participant-guide.html"

frame_hub = FrameHub()
runtime = RobotRuntime(frame_hub)
real_runtime = RealRobotRuntime(frame_hub, runtime)
hardware_setup = HardwareSetupManager()


def make_viewer(model, data, source, playback_speed):
    return PortalViewer(
        model,
        data,
        frame_hub,
        source,
        playback_speed=playback_speed,
    )


workflow = WorkflowManager(runtime, frame_hub, make_viewer, real_runtime)
chat = ChatAgent(workflow)
workflow.set_chat_handler(chat.handle)
workflow.set_result_handler(chat.report_physical_result)


def require_physical_setup_idle():
    if not real_runtime.connected:
        raise RuntimeError("connect the physical SO-101 first")
    if workflow.snapshot()["busy"]:
        raise RuntimeError("wait for the current workflow step to finish")
    if real_runtime.autonomous_active:
        raise RuntimeError("wait for physical motion to stop")


def invalidate_physical_scene_after_setup_change():
    workflow.clear_memory(include_calibration=True)
    chat.cancel_pending()


@asynccontextmanager
async def lifespan(app):
    del app
    frame_hub.start()
    runtime.start()
    yield
    workflow.close()
    if real_runtime.connected:
        real_runtime.disconnect()
    runtime.stop()
    frame_hub.stop()


app = FastAPI(
    title="Agentic Robotics Workshop Portal",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ManualControl(BaseModel):
    joint: str
    direction: int = Field(ge=-1, le=1)
    active: bool


class ActionRequest(BaseModel):
    target: str | None = None
    destination: str | None = None
    labels: list[dict[str, str]] | None = None


class ClearMemoryRequest(BaseModel):
    include_calibration: bool = False


class RealRobotConnection(BaseModel):
    port: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    camera_index: int = Field(ge=0)
    width: int = Field(default=480, ge=160, le=1920)
    height: int = Field(default=640, ge=120, le=1920)
    fps: int = Field(default=30, ge=5, le=60)
    rotation: int = 90


class ArmRequest(BaseModel):
    armed: bool


class TeleopStart(BaseModel):
    leader_port: str = Field(min_length=1)
    leader_id: str = Field(min_length=1)


class ReplayRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class HardwareDiscoveryRequest(BaseModel):
    include_cameras: bool = True


class BoardPoseRequest(BaseModel):
    name: Literal["survey_left", "survey_center", "survey_right"]
    board_origin_x: float = Field(ge=-1.0, le=1.0)
    board_origin_y: float = Field(ge=-1.0, le=1.0)
    board_origin_z: float = Field(default=0.0, ge=-0.2, le=0.5)
    board_yaw_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)


class WorkspaceRequest(BaseModel):
    minimum: list[float] = Field(min_length=3, max_length=3)
    maximum: list[float] = Field(min_length=3, max_length=3)
    table_minimum: list[float] = Field(min_length=2, max_length=2)
    table_maximum: list[float] = Field(min_length=2, max_length=2)
    destinations: dict[str, list[float]]
    gripper_open_percent: float = Field(ge=1.0, le=99.0)
    gripper_closed_percent: float = Field(ge=1.0, le=99.0)
    base_frame_registration_confirmed: bool = False
    kinematic_model_confirmed: bool = False


class PhysicalApprovalRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=80)


class PlanApprovalRequest(PhysicalApprovalRequest):
    plan_id: str = Field(min_length=1, max_length=300)


class LLMSettings(BaseModel):
    provider: Literal["openrouter", "openai", "anthropic", "glm", "custom"]
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=500)
    base_url: str | None = Field(default=None, max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    settings: LLMSettings


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/guide")
def guide():
    return FileResponse(GUIDE_PATH)


@app.get("/api/state")
def state():
    paths = workflow.paths()
    artifacts, _ = artifact_state(paths)
    return {
        "workflow": workflow.snapshot(),
        "readiness": readiness_state(paths),
        "scene": scene_state(paths),
        "artifacts": artifacts,
        "stream": frame_hub.metadata(),
        "robot": real_runtime.snapshot(),
        "hardware": hardware_setup.snapshot(),
        "chat": chat.snapshot(),
    }


@app.get("/api/frame/{camera}")
def frame(camera: str):
    if camera not in {"main", "wrist"}:
        raise HTTPException(status_code=404, detail="unknown camera")
    image = frame_hub.frame(camera)
    if image is None:
        raise HTTPException(status_code=503, detail="renderer is warming up")
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/stream/{camera}")
async def stream(camera: str, request: Request):
    if camera not in {"main", "wrist"}:
        raise HTTPException(status_code=404, detail="unknown camera")

    async def frames():
        sequence = -1
        while not await request.is_disconnected():
            image, next_sequence = frame_hub.frame_with_sequence(camera)
            if image is None or next_sequence == sequence:
                await asyncio.sleep(0.01)
                continue
            sequence = next_sequence
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(image)}\r\n\r\n".encode()
                + image
                + b"\r\n"
            )

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/artifact/{artifact_id}")
def artifact(artifact_id: str):
    _, paths = artifact_state(workflow.paths())
    path = paths.get(artifact_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact is not available")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/manual/control")
def manual_control(control: ManualControl):
    try:
        active_runtime = real_runtime if real_runtime.connected else runtime
        active_runtime.set_control(control.joint, control.direction, control.active)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"ok": True}


@app.post("/api/manual/release")
def release_controls():
    runtime.release_all()
    real_runtime.release_all()
    return {"ok": True}


@app.post("/api/manual/heartbeat")
def refresh_manual_control_lease():
    real_runtime.refresh_manual_control_lease()
    return {"ok": True}


@app.post("/api/manual/home")
def home():
    if real_runtime.connected:
        raise HTTPException(
            status_code=409,
            detail="home pose is simulation-only; move the real robot deliberately",
        )
    try:
        runtime.home()
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"ok": True}


@app.post("/api/manual/reset")
def reset():
    if real_runtime.connected:
        raise HTTPException(
            status_code=409,
            detail="reset world is not available for a physical robot",
        )
    try:
        runtime.reset()
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"ok": True}


@app.post("/api/action/{action}")
def action(action: str, request: ActionRequest | None = None):
    if real_runtime.connected and action not in {
        "scan_empty",
        "scan_scene",
        "reconstruct_scene",
        "label_manual",
        "plan",
    }:
        raise HTTPException(
            status_code=409,
            detail="this action is not available in physical mode",
        )
    payload = request.model_dump(exclude_none=True) if request else {}
    try:
        if real_runtime.connected:
            real_runtime.lock_manual_control()
        workflow.start(action, payload)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"accepted": True, "action": action}


@app.post("/api/memory/clear")
def clear_memory(request: ClearMemoryRequest):
    try:
        workflow.clear_memory(request.include_calibration)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "cleared": True,
        "include_calibration": request.include_calibration,
    }


@app.post("/api/chat")
def send_chat(request: ChatRequest):
    try:
        settings = request.settings.model_dump()
        settings["model"] = settings["model"].strip()
        settings["api_key"] = settings["api_key"].strip()
        if settings["base_url"]:
            settings["base_url"] = settings["base_url"].strip()
        if not settings["model"]:
            raise ValueError("choose a model before sending a command")
        if settings["provider"] != "custom" and not settings["api_key"]:
            raise ValueError("enter an API key for the selected provider")
        if settings["provider"] == "custom" and not settings["base_url"]:
            raise ValueError("enter a base URL for the custom provider")
        if real_runtime.connected:
            real_runtime.lock_manual_control()
        chat.submit(request.message, settings)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"accepted": True}


@app.post("/api/chat/clear")
def clear_chat():
    if workflow.snapshot()["busy"]:
        raise HTTPException(
            status_code=409,
            detail="wait for the current copilot turn to finish",
        )
    try:
        chat.clear()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"cleared": True}


@app.post("/api/robot/connect")
def connect_real_robot(connection: RealRobotConnection):
    if connection.rotation not in {0, 90, 180, 270}:
        raise HTTPException(
            status_code=422,
            detail="camera rotation must be 0, 90, 180, or 270 degrees counter-clockwise",
        )
    if workflow.snapshot()["busy"]:
        raise HTTPException(status_code=409, detail="wait for the current workflow step")
    try:
        workflow.invalidate_plan()
        chat.cancel_pending()
        real_runtime.connect(**connection.model_dump())
        workflow.clear_memory(include_calibration=True)
    except (RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"connected": True, "robot": real_runtime.snapshot()}


@app.post("/api/hardware/discover")
def discover_hardware(request: HardwareDiscoveryRequest):
    if real_runtime.connected:
        raise HTTPException(
            status_code=409,
            detail="disconnect the physical robot before scanning hardware",
        )
    return hardware_setup.discover(request.include_cameras)


@app.post("/api/robot/arm")
def arm_real_robot(request: ArmRequest):
    try:
        if request.armed:
            workflow.invalidate_plan()
        real_runtime.arm() if request.armed else real_runtime.disarm()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"armed": real_runtime.armed}


@app.post("/api/robot/teleop/start")
def start_teleop(request: TeleopStart):
    try:
        workflow.invalidate_plan()
        chat.cancel_pending()
        real_runtime.start_teleop(request.leader_port, request.leader_id)
    except (RuntimeError, ValueError, OSError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"teleop": True}


@app.post("/api/robot/teleop/stop")
def stop_teleop():
    try:
        saved = real_runtime.stop_teleop()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"teleop": False, "saved": saved}


@app.post("/api/robot/teleop/record/start")
def start_teleop_recording():
    try:
        return real_runtime.start_recording()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/robot/teleop/record/stop")
def stop_teleop_recording():
    try:
        return real_runtime.stop_recording()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/robot/teleop/recordings")
def teleop_recordings():
    return {"recordings": real_runtime.list_recordings()}


@app.delete("/api/robot/teleop/recordings/{name}")
def delete_teleop_recording(name: str):
    try:
        return real_runtime.delete_recording(name)
    except RuntimeError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/robot/teleop/replay")
def replay_teleop(request: ReplayRequest):
    try:
        result = real_runtime.replay(request.name)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"replayed": True, "result": result}


@app.post("/api/robot/calibration/intrinsics/sample")
def capture_intrinsic_sample():
    try:
        require_physical_setup_idle()
        return real_runtime.add_intrinsic_sample()
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/robot/calibration/intrinsics/solve")
def solve_intrinsics():
    try:
        require_physical_setup_idle()
        result = real_runtime.solve_intrinsics()
        invalidate_physical_scene_after_setup_change()
        return result
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/robot/calibration/travel")
def record_travel_pose():
    try:
        require_physical_setup_idle()
        result = real_runtime.record_travel_pose()
        invalidate_physical_scene_after_setup_change()
        return result
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/robot/calibration/viewpoint")
def record_viewpoint(request: BoardPoseRequest):
    payload = request.model_dump()
    name = payload.pop("name")
    try:
        require_physical_setup_idle()
        result = real_runtime.record_viewpoint(name, **payload)
        invalidate_physical_scene_after_setup_change()
        return result
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/robot/calibration/workspace")
def save_workspace(request: WorkspaceRequest):
    try:
        require_physical_setup_idle()
        result = real_runtime.save_workspace(request.model_dump())
        invalidate_physical_scene_after_setup_change()
        return result
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.delete("/api/robot/calibration")
def clear_physical_calibration():
    try:
        require_physical_setup_idle()
        real_runtime.disarm()
        real_runtime.workspace.clear()
        invalidate_physical_scene_after_setup_change()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"cleared": True}


@app.post("/api/robot/approve-scan")
def approve_physical_scan(request: PhysicalApprovalRequest):
    if request.confirmation != "APPROVE ONE SCAN":
        raise HTTPException(status_code=400, detail="scan confirmation did not match")
    try:
        workflow.invalidate_plan()
        real_runtime.approve_scan()
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"approved": True, "scope": "one_scan"}


@app.post("/api/robot/approve-plan")
def approve_physical_plan(request: PlanApprovalRequest):
    if request.confirmation != "APPROVE PHYSICAL MOTION":
        raise HTTPException(status_code=400, detail="motion confirmation did not match")
    try:
        workflow.require_current_physical_plan(request.plan_id)
        real_runtime.approve_plan(request.plan_id)
        workflow.start("execute_real", {"plan_id": request.plan_id})
    except (RuntimeError, ValueError) as error:
        real_runtime.disarm()
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"accepted": True, "plan_id": request.plan_id}


@app.post("/api/robot/disconnect")
def disconnect_real_robot():
    workflow.invalidate_plan()
    chat.cancel_pending()
    try:
        real_runtime.disconnect()
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"connected": False}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        access_log=False,
        timeout_graceful_shutdown=2,
    )
