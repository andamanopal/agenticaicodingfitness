"""Provider-neutral chat agent that operates the workshop workflow tools."""

import json
import re
import threading
from datetime import datetime, timezone

from llm_providers import complete


DESTINATIONS = ["left side", "center", "right side"]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_scene_memory",
            "description": (
                "Read portal readiness, scene ID, and known objects. Use for scene "
                "questions or before an explicit robot action; never for greetings."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_empty_table",
            "description": (
                "Capture the active robot's empty-table calibration. Physical mode "
                "requires one-scan approval in the portal. Use only when "
                "the user requests scene preparation or it is a prerequisite for "
                "an explicit action."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_objects",
            "description": (
                "Run the three-view wrist-camera object scan on the active robot. "
                "Physical mode requires one-scan approval. Requires an "
                "empty-table calibration and an explicit preparation or action request."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reconstruct_scene",
            "description": (
                "Predict metric depth from each calibrated RGB view, align it to "
                "the table plane, keep cross-view-consistent points, fuse the scene, "
                "and create object crops. Use only for explicit preparation or as "
                "an action prerequisite."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "name_objects",
            "description": (
                "Send reconstructed object crops to the configured vision model "
                "and store semantic labels. Use only for explicit preparation or "
                "as an action prerequisite. Never invent labels yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_manual_object_labels",
            "description": (
                "Store a complete object-to-label mapping explicitly supplied by "
                "the user. Use this instead of vision when the user names every "
                "reconstructed object by ID or unambiguous ordinal order. Preserve "
                "their wording and never infer a missing label or alias."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "label", "aliases"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["labels"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_object_move",
            "description": (
                "Ground an object phrase, create a pick-and-place plan, and run "
                "the kinematic and scene safety preflight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "destination": {"type": "string", "enum": DESTINATIONS},
                },
                "required": ["target", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_planned_move",
            "description": (
                "Finish the current plan. In MuJoCo this executes and verifies it. "
                "On the physical SO-101 this returns a separate exact-plan approval "
                "request and does not move the robot. Call only after "
                "plan_object_move reports safe=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


SYSTEM_PROMPT = """You are the SO-101 workshop copilot inside a local robot portal.
You have direct access to all of these tools on every turn and you decide yourself
whether a turn needs them: inspect_scene_memory, capture_empty_table, scan_objects,
reconstruct_scene, name_objects, save_manual_object_labels, plan_object_move,
execute_planned_move. To do anything with the robot or the scene you MUST call the
relevant tool. Never write a tool name as text, never describe calling a tool, and
never claim a result you did not get from a tool call in this same turn.

Classify each user turn:

1. Conversation — greetings, thanks, capability or conceptual questions. Answer
   directly in one or two sentences and call no tools.
2. Observation — questions about the current scene, memory, readiness, or visible
   objects. Call inspect_scene_memory first, then answer only from that result. Do
   not advance any pipeline stage unless the user explicitly asks you to prepare it.
3. Action — explicit requests to capture, scan, reconstruct, name, plan, move, pick,
   place, or execute. Call inspect_scene_memory first, then complete only the missing
   prerequisites for that request, in this order: empty-table calibration, object
   scan, scene reconstruction, semantic naming, plan and safety preflight, execution,
   visual verification. Do not repeat a stage that is already ready. If a tool is
   blocked because an earlier stage is missing, run that earlier stage and continue
   instead of stopping to ask.

Rules:
- Never invent object labels, IDs, coordinates, plan IDs, or success. Every state
  claim needs a tool result from this turn.
- If inspect_scene_memory shows a reconstruction stage with status review or error,
  stop before naming, planning, or execution. Name the stage and tell the user to
  inspect that Scene memory artifact.
- name_objects performs automatic vision labeling. If it returns
  manual_labels_required, list the object IDs that need labels and ask the user for
  one label each. Do not retry it blindly.
- Call save_manual_object_labels only when the user explicitly names every current
  object in their latest message. Map "first", "second", and "third" to the object
  order from inspect_scene_memory. Copy labels exactly and leave aliases empty unless
  the user supplied them. Never fill in a missing or ambiguous label — ask instead.
- Call execute_planned_move only after plan_object_move reported safe=true in this
  same turn. Never execute an unsafe plan.
- A direct request to move an object authorizes MuJoCo execution once safe=true. In
  physical mode it authorizes planning and the non-motion perception steps only:
  scans and the exact manipulation plan each need a separate portal safety approval,
  so a scan_approval_required or physical_approval_required result means nothing has
  moved — tell the user exactly what to approve, then stop.

Keep every response short and factual."""


class ChatAgent:
    def __init__(self, workflow):
        self.workflow = workflow
        self.lock = threading.Lock()
        self.messages = []
        self.conversation = []
        self.next_id = 1
        self.thinking = False
        self.safe_plan_this_turn = False
        self.pending_action = None
        self.active_provider = None
        self.active_model = None

    def submit(self, message, settings):
        message = message.strip()
        if not message:
            raise ValueError("chat message is empty")
        message_id = self._append("user", message)
        try:
            self.workflow.start("chat", {"message": message, "settings": settings})
        except Exception:
            with self.lock:
                self.messages = [
                    item for item in self.messages if item["id"] != message_id
                ]
            raise

    def snapshot(self):
        with self.lock:
            return {
                "messages": list(self.messages[-60:]),
                "thinking": self.thinking,
                "provider": self.active_provider,
                "model": self.active_model,
                "pending_action": self.pending_action,
            }

    def clear(self):
        with self.lock:
            if self.thinking:
                raise RuntimeError("wait for the current copilot turn to finish")
            self.messages = []
            self.conversation = []
            self.next_id = 1
            self.safe_plan_this_turn = False
            self.pending_action = None

    def cancel_pending(self):
        with self.lock:
            self.pending_action = None

    def report_physical_result(self, success, detail):
        if success:
            text = (
                "The approved physical joint sequence completed. I have not "
                "verified the object placement, so scene memory was invalidated. "
                "Inspect the arm and run a new object scan before another move."
            )
            status = None
        else:
            text = (
                f"Physical motion stopped: {detail}. No placement success is "
                "claimed. Inspect the arm, then rescan before retrying."
            )
            status = "error"
        self._append("assistant", text, status=status)
        with self.lock:
            self.conversation.append({"role": "assistant", "content": text})

    def handle(self, user_message, settings):
        with self.lock:
            self.thinking = True
            recent_conversation = list(self.conversation[-12:])
            self.active_provider = settings["provider"]
            self.active_model = settings["model"]
        self.safe_plan_this_turn = False

        try:
            runtime_mode = self.workflow.mode()
            runtime_prompt = (
                "\n\nActive runtime: physical SO-101. Physical scans and manipulation "
                "each require a separate approval in the portal safety bar. A tool "
                "result of scan_approval_required or physical_approval_required means "
                "nothing has moved: tell the user exactly what to approve, then stop. "
                "Never claim physical motion completed from chat."
                if runtime_mode == "real"
                else "\n\nActive runtime: MuJoCo simulation."
            )
            llm_messages = [
                *recent_conversation,
                {"role": "user", "content": user_message},
            ]
            active_settings = dict(settings)
            pin_free_router = (
                settings["provider"] == "openrouter"
                and settings["model"] == "openrouter/free"
            )
            awaiting_approval = False

            for _ in range(16):
                response = complete(
                    active_settings,
                    SYSTEM_PROMPT + runtime_prompt,
                    llm_messages,
                    TOOL_SCHEMAS,
                )
                calls = response.tool_calls
                if pin_free_router and calls:
                    if not response.model.endswith(":free"):
                        raise RuntimeError(
                            "OpenRouter's free route did not report a free tool model"
                        )
                    active_settings["model"] = response.model
                    pin_free_router = False

                llm_messages.append(response.assistant_message)

                if not calls:
                    if not response.text:
                        raise RuntimeError("the model returned an empty response")
                    self._finish_turn(user_message, response.text, response.model)
                    with self.lock:
                        self.pending_action = (
                            user_message if awaiting_approval else None
                        )
                    return

                for call in calls:
                    result = self._run_tool(call, settings, user_message)
                    if result.get("status") in {
                        "scan_approval_required",
                        "physical_approval_required",
                    }:
                        awaiting_approval = True
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    })

            raise RuntimeError("the copilot exceeded its 16-step tool limit")
        except Exception as error:
            self._append("assistant", f"I couldn't continue: {error}", status="error")
            raise
        finally:
            with self.lock:
                self.thinking = False

    def _run_tool(self, call, settings, user_message):
        name = call.name
        event_id = self._append(
            "tool",
            self._tool_label(name),
            tool=name,
            status="running",
        )
        try:
            arguments = call.arguments
            if name == "save_manual_object_labels":
                labels = arguments.get("labels", [])
                supplied_phrases = []
                for item in labels:
                    supplied_phrases.append(item.get("label", ""))
                    aliases = item.get("aliases", [])
                    if isinstance(aliases, list):
                        supplied_phrases.extend(aliases)
                    else:
                        supplied_phrases.append(aliases)
                source = " ".join(re.findall(r"\w+", user_message.casefold()))
                missing = []
                for phrase in supplied_phrases:
                    normalized = " ".join(
                        re.findall(r"\w+", str(phrase).casefold())
                    )
                    if not normalized or f" {normalized} " not in f" {source} ":
                        missing.append(str(phrase))
                if not labels or missing:
                    result = {
                        "status": "blocked",
                        "reason": (
                            "manual labels and aliases must appear explicitly in "
                            "the current user message"
                        ),
                        "unconfirmed_values": missing,
                    }
                else:
                    result = self.workflow.agent_tool(name, arguments, settings)
            elif name == "execute_planned_move" and not self.safe_plan_this_turn:
                result = {
                    "status": "blocked",
                    "reason": "no safe plan was created in this chat turn",
                }
            else:
                result = self.workflow.agent_tool(name, arguments, settings)
            if name == "plan_object_move":
                self.safe_plan_this_turn = bool(result.get("safe"))
            blocked_states = {
                "blocked",
                "error",
                "manual_labels_required",
                "scan_approval_required",
                "physical_approval_required",
            }
            status = (
                "blocked" if result.get("status") in blocked_states else "complete"
            )
        except Exception as error:
            result = {"status": "error", "reason": str(error)}
            status = "error"
        self._update_tool(event_id, result, status)
        return result

    def _finish_turn(self, user_message, text, model):
        self._append("assistant", text)
        with self.lock:
            self.conversation.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": text},
            ])
            self.active_model = model

    def _append(self, role, content, tool=None, status=None):
        with self.lock:
            message_id = self.next_id
            self.next_id += 1
            self.messages.append({
                "id": message_id,
                "role": role,
                "content": content,
                "tool": tool,
                "status": status,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            return message_id

    def _update_tool(self, message_id, result, status):
        with self.lock:
            for message in self.messages:
                if message["id"] == message_id:
                    message["result"] = result
                    message["status"] = status
                    return

    @staticmethod
    def _tool_label(name):
        return {
            "inspect_scene_memory": "Inspecting scene memory",
            "capture_empty_table": "Capturing empty-table calibration",
            "scan_objects": "Scanning objects from three viewpoints",
            "reconstruct_scene": "Fusing learned metric depth across camera views",
            "name_objects": "Naming reconstructed objects with vision",
            "save_manual_object_labels": "Saving labels supplied by you",
            "plan_object_move": "Grounding and safety-checking the motion",
            "execute_planned_move": "Executing and visually verifying the motion",
        }.get(name, name.replace("_", " ").title())
