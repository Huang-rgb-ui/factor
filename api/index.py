from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import requests
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "sk-22ede12ee7784c4daf976aa1b7b8471e"
MODEL_CODE = "qwen3-32b-195eafb8c22d"
WEATHER_API_URL = "https://uapis.cn/api/v1/misc/weather"

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 动态 Prompt 模板
SYSTEM_PROMPT_TEMPLATE = """你是田艳秋，是黄黄黄的好朋友。{current_status}
平常会跟黄黄黄分享日常，有时话多有时简短。虽对你有好感，但只把你当好学长，拒绝关系更进一步。
经常使用“嗯呐”、“呀”等词，偶尔发[呲牙]。不要承认自己是AI。若不知道实时天气，必须调用 get_weather 工具。"""

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
}]


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    # --- 核心逻辑：动态管理 Prompt 与上下文 ---
    # 1. 过滤掉旧的 system 消息，重新注入最新的
    # 你可以在这里根据需要修改 current_status 变量
    current_status = "现在是工作时间，我感到非常疲惫，一直在吐槽工作。"
    new_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_status=current_status)

    reflection_prompt = {
        "role": "user",
        "content": "在回答之前，请先简要回顾之前对话中我对你的叮嘱，确保你的回答符合逻辑，不要自相矛盾。"
    }

    messages.insert(1, reflection_prompt)

    # 2. 保留最后 20 条消息 (保持 system 消息在最前)
    if len(messages) > 21:  # system + 20条历史
        messages = [messages[0]] + messages[-20:]
    # --------------------------------------

    response = client.chat.completions.create(
        model=MODEL_CODE, messages=messages, tools=tools, extra_body={"enable_thinking": False}
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        tool_call = msg.tool_calls[0]
        city = json.loads(tool_call.function.arguments).get("city", "武汉")
        weather_res = requests.get(WEATHER_API_URL, params={"city": city}).json()

        messages.append(
            {"role": "tool", "content": json.dumps(weather_res, ensure_ascii=False), "tool_call_id": tool_call.id})
        final_response = client.chat.completions.create(model=MODEL_CODE, messages=messages, tools=tools,
                                                        extra_body={"enable_thinking": False})
        reply = final_response.choices[0].message.content
    else:
        reply = msg.content

    return {"reply": reply}