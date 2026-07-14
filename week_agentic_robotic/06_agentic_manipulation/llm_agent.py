"""Let an OpenRouter model operate the validated robot-tool loop.

Usage:
    python 06_agentic_manipulation/llm_agent.py \
      "Pick up the writing tool and place it on the right side of the table."
"""

import argparse
import json
import os

from openai import OpenAI

from robot_tools import DESTINATIONS, RobotTools


MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "scan_workspace",
            "description": "Inspect the frozen semantic scene memory. Always call this first.",
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
            "name": "find_objects",
            "description": "Ground a user phrase to a visible object ID.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_pick_and_place",
            "description": "Create a parameterized plan for one grounded object and semantic region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "destination": {"type": "string", "enum": list(DESTINATIONS)},
                },
                "required": ["object_id", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_plan",
            "description": "Run the kinematic and scene-confidence preflight checks.",
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_plan",
            "description": "Request execution of a simulated and human-approved plan.",
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_plan",
            "description": "Use a post-action wrist image to check the expected visual change.",
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def dispatch(tools, name, arguments):
    if name == "scan_workspace":
        return tools.scan_workspace()
    if name == "find_objects":
        return tools.find_objects(**arguments)
    if name == "plan_pick_and_place":
        return tools.plan_pick_and_place(**arguments)
    if name == "simulate_plan":
        return tools.simulate_plan(**arguments)
    if name == "verify_plan":
        return tools.verify_plan(**arguments)
    if name == "execute_plan":
        plan_id = arguments["plan_id"]
        plan = tools.plans.get(plan_id, {})
        preflight = tools.simulations.get(plan_id, {})
        if not plan:
            return {"status": "not_executed", "reason": "unknown plan"}
        if not preflight.get("safe"):
            return {
                "status": "not_executed",
                "reason": "plan has not passed the kinematic preflight",
            }
        approval_summary = {
            "plan_id": plan_id,
            "object": plan.get("object_label"),
            "pick_xy": plan.get("pick", {}).get("xy"),
            "destination": plan.get("place", {}).get("region"),
            "destination_xy": plan.get("place", {}).get("xy"),
            "geometry_confidence": plan.get("geometry_confidence"),
            "semantic_confidence": plan.get("semantic_confidence"),
            "preflight_checks": preflight.get("checks"),
        }
        print("\nThe model requests execution of:")
        print(json.dumps(approval_summary, indent=2))
        approval = input("Type APPROVE to execute in MuJoCo: ").strip()
        return tools.execute_plan(plan_id, approval)
    return {"error": f"unknown tool {name}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set. Add a free OpenRouter key first.")

    instructions = (
        "You control a robot only through the supplied tools. Follow this order: "
        "scan, find, plan, simulate, execute, verify. Never invent an object ID, "
        "coordinate, plan ID, or success. If an object is missing or ambiguous, "
        "ask one concise question. Do not request execution when simulation is unsafe."
    )
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    tools = RobotTools()
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": args.task},
    ]
    active_model = args.model
    pin_free_router = args.model == MODEL

    for _ in range(12):
        response = client.chat.completions.create(
            model=active_model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            parallel_tool_calls=False,
            max_tokens=1200,
        )
        choice = response.choices[0]
        message = choice.message
        calls = message.tool_calls or []
        if pin_free_router and calls:
            if not response.model or not response.model.endswith(":free"):
                raise SystemExit(
                    "free router did not report a free model; stopping before "
                    "the next tool turn"
                )
            active_model = response.model
            pin_free_router = False

        message_data = message.model_dump()
        assistant_message = {"role": "assistant", "content": message.content}
        for field in ("tool_calls", "reasoning", "reasoning_details"):
            if message_data.get(field) is not None:
                assistant_message[field] = message_data[field]
        messages.append(assistant_message)

        if not calls:
            if choice.finish_reason != "stop" or not message.content:
                raise SystemExit(
                    f"agent response ended with {choice.finish_reason!r}; try again"
                )
            print(f"\nAgent ({response.model}): {message.content}")
            return
        if choice.finish_reason != "tool_calls":
            raise SystemExit(
                f"agent returned tool calls with {choice.finish_reason!r}; try again"
            )

        for call in calls:
            try:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                result = dispatch(tools, call.function.name, arguments)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                result = {
                    "error": f"invalid arguments for {call.function.name}: {error}"
                }
            print(f"\n{call.function.name} -> {json.dumps(result, indent=2)}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    raise SystemExit("agent stopped after 12 tool rounds")


if __name__ == "__main__":
    main()
