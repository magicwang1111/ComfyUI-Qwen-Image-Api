import base64
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests


QWEN_MODELS = ["qwen-image-3.0-pro", "qwen-image-3.0", "qwen-image-2.1-pro", "qwen-image-2.1-turbo"]
WAN_MODELS = ["wan2.7-image-pro", "wan2.7-image"]
SYNC_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
ASYNC_PATH = "/api/v1/services/aigc/image-generation/generation"
OPENAI_PATH = "/compatible-mode/v1/images/generations"
CONFIG_PATH = Path(__file__).with_name("config.local.json")


def connection(base_url="", api_key=""):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    host = (base_url.strip() or os.getenv("DASHSCOPE_BASE_URL", "").strip()
            or config.get("base_url", "") or "https://dashscope.aliyuncs.com").rstrip("/")
    key = api_key.strip() or os.getenv("DASHSCOPE_API_KEY", "").strip() or config.get("api_key", "")
    if not key:
        raise ValueError("请设置 DASHSCOPE_API_KEY、config.local.json 或节点 api_key。")
    if "://" not in host:
        host = "https://" + host
    for suffix in (SYNC_PATH, ASYNC_PATH, OPENAI_PATH, "/compatible-mode/v1", "/api/v1"):
        if host.endswith(suffix):
            host = host[:-len(suffix)]
            break
    parsed = urlsplit(host)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path):
        raise ValueError("base_url 请填写业务空间 Host、DashScope /api/v1 或 /compatible-mode/v1 地址。")
    return host, key


def integer_range(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} 必须为 {low} 至 {high} 的整数。")


def output_size(size, minimum, maximum, separator="*", presets=()):
    value = size.strip()
    if not value or value.lower() == "auto":
        return None
    if value.upper() in presets:
        return value.upper()
    match = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", value)
    if not match:
        raise ValueError("size 请填写 auto、宽*高（如 1024*1024）或该模型支持的 1K/2K/4K。")
    width, height = map(int, match.groups())
    if min(width, height) <= 0 or not minimum <= width * height <= maximum or max(width, height) > 8 * min(width, height):
        raise ValueError(f"size 总像素须在 {minimum} 至 {maximum}，宽高比须在 1:8 至 8:1。")
    return f"{width}{separator}{height}"


def dashscope_body(model, prompt, images, parameters):
    content = [{"image": image} for image in images] + [{"text": prompt}]
    return {"model": model, "input": {"messages": [{"role": "user", "content": content}]}, "parameters": parameters}


def qwen_body(model, prompt, images, protocol="openai", size="auto", n=1, seed=-1,
              prompt_extend=True, prompt_extend_mode="direct", enable_thinking=True,
              watermark=False, negative_prompt="", response_format="url"):
    if model not in QWEN_MODELS or protocol not in ("openai", "dashscope"):
        raise ValueError("不支持的千问模型或协议。")
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")
    integer_range("n", n, 1, 6)
    integer_range("seed", seed, -1, 2147483647)
    is_v3 = model.startswith("qwen-image-3.0")
    limit = 3 if is_v3 else 10
    if len(images) > limit:
        raise ValueError(f"{model} 最多支持 {limit} 张参考图，当前 {len(images)} 张。")
    if prompt_extend_mode not in ("direct", "agent") or (images and prompt_extend_mode == "agent"):
        raise ValueError("agent 改写模式仅支持文生图，图像编辑请使用 direct。")
    if response_format not in ("url", "b64_json") or (response_format == "b64_json" and not is_v3):
        raise ValueError("b64_json 仅支持千问 3.0 系列同步调用。")
    if negative_prompt.strip() and not is_v3:
        raise ValueError("negative_prompt 仅支持千问 3.0 系列。")
    params = {"n": n, "prompt_extend": prompt_extend, "prompt_extend_mode": prompt_extend_mode,
              "enable_thinking": enable_thinking, "watermark": watermark, "response_format": response_format}
    resolution = output_size(size, 512**2, 2048**2, "x" if protocol == "openai" else "*")
    if resolution:
        params["size"] = resolution
    if seed >= 0:
        params["seed"] = seed
    if negative_prompt.strip():
        params["negative_prompt"] = negative_prompt
    if protocol == "openai":
        body = {"model": model, "prompt": prompt, **params}
        if images:
            body["image"] = images
        return body
    return dashscope_body(model, prompt, images, params)


def wan_body(model, prompt, images, size="2K", n=1, seed=-1, enable_sequential=False,
             thinking_mode=True, watermark=False, bbox_list="", color_palette=""):
    if model not in WAN_MODELS:
        raise ValueError("不支持的万相模型。")
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")
    if len(images) > 9:
        raise ValueError(f"万相最多支持 9 张参考图，当前 {len(images)} 张。")
    integer_range("n", n, 1, 12 if enable_sequential else 4)
    integer_range("seed", seed, -1, 2147483647)
    supports_4k = model == "wan2.7-image-pro" and not images and not enable_sequential
    presets = ("1K", "2K", "4K") if supports_4k else ("1K", "2K")
    if size.strip().upper() == "4K" and not supports_4k:
        raise ValueError("4K 仅支持 wan2.7-image-pro 无参考图且关闭组图的文生图。")
    resolution = output_size(size, 768**2, 4096**2 if supports_4k else 2048**2, presets=presets)
    params = {"n": n, "enable_sequential": enable_sequential, "watermark": watermark}
    if resolution:
        params["size"] = resolution
    if not images and not enable_sequential:
        params["thinking_mode"] = thinking_mode
    if seed >= 0:
        params["seed"] = seed
    if bbox_list.strip():
        boxes = json.loads(bbox_list)
        if not images or not isinstance(boxes, list) or len(boxes) != len(images):
            raise ValueError("bbox_list 必须为与参考图数量一致的 JSON 数组。")
        for group in boxes:
            if not isinstance(group, list) or len(group) > 2:
                raise ValueError("每张图片最多支持两个边界框。")
            for box in group:
                if (not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box)
                        or min(box) < 0 or box[0] >= box[2] or box[1] >= box[3]):
                    raise ValueError("边界框格式应为 [x1,y1,x2,y2]，使用原图非负整数坐标。")
        params["bbox_list"] = boxes
    if color_palette.strip():
        if enable_sequential:
            raise ValueError("组图模式不支持 color_palette。")
        palette = json.loads(color_palette)
        if not isinstance(palette, list) or not 3 <= len(palette) <= 10:
            raise ValueError("color_palette 必须包含 3 至 10 种颜色。")
        total = Decimal(0)
        for color in palette:
            if (not isinstance(color, dict) or not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color.get("hex", "")))
                    or not re.fullmatch(r"\d{1,3}\.\d{2}%", str(color.get("ratio", "")))):
                raise ValueError('配色格式为 {"hex":"#C2D1E6","ratio":"25.00%"}。')
            total += Decimal(color["ratio"][:-1])
        if total != Decimal("100.00"):
            raise ValueError("color_palette 比例总和必须为 100.00%。")
        params["color_palette"] = palette
    return dashscope_body(model, prompt, images, params)


def request_json(method, url, key, timeout, body=None, asynchronous=False):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if asynchronous:
        headers["X-DashScope-Async"] = "enable"
    try:
        with requests.request(method, url, headers=headers, json=body, timeout=(min(30, timeout), timeout)) as response:
            request_id = response.headers.get("x-request-id", "")
            try:
                data = response.json()
            except ValueError:
                raise RuntimeError(f"接口返回非 JSON 内容，HTTP {response.status_code}，request_id={request_id}") from None
            if not isinstance(data, dict):
                raise RuntimeError(f"接口返回格式错误，request_id={request_id}")
            request_id = data.get("request_id") or request_id
            error = data.get("error") or {}
            if not response.ok or data.get("code") or error:
                details = error if isinstance(error, dict) and error else data
                message = str(details.get("message", "请求失败")).replace(key, "[REDACTED]")
                raise RuntimeError(f"HTTP {response.status_code} {details.get('code', '')}: {message}; request_id={request_id}")
            data["request_id"] = request_id
            return data
    except requests.RequestException as exc:
        raise RuntimeError(f"API 连接失败（{type(exc).__name__}）；未自动重新提交，请检查连接和超时设置。") from None


def generate(body, base_url="", api_key="", protocol="dashscope", timeout=600, check_interrupt=lambda: None, resume_task_id=""):
    host, key = connection(base_url, api_key)
    check_interrupt()
    path = OPENAI_PATH if protocol == "openai" else ASYNC_PATH if protocol == "async" else SYNC_PATH
    task_id = resume_task_id.strip()
    if not task_id:
        data = request_json("POST", host + path, key, timeout, body, protocol == "async")
        if protocol != "async":
            return data
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"异步接口未返回 task_id；request_id={data.get('request_id', '')}")
    print(f"[Qwen/Wan Image API] task_id={task_id}")
    deadline = time.monotonic() + timeout
    while True:
        check_interrupt()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"等待任务超时，task_id={task_id}；云端任务可能仍在运行，可通过 resume_task_id 继续查询。")
        try:
            data = request_json("GET", host + "/api/v1/tasks/" + quote(task_id, safe=""), key, min(60, remaining))
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; task_id={task_id}，可通过 resume_task_id 继续查询。") from None
        output = data.get("output", {})
        status = output.get("task_status")
        if status == "SUCCEEDED":
            return data
        if status not in ("PENDING", "RUNNING"):
            raise RuntimeError(f"任务 {task_id}: {status}; {output.get('code', '')} {output.get('message', '')}; request_id={data.get('request_id', '')}")
        for _ in range(5):
            check_interrupt()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1, remaining))


def result_images(data):
    images = []
    for item in data.get("data") or []:
        if item.get("b64_json"):
            images.append("data:image/png;base64," + item["b64_json"])
        elif item.get("url"):
            images.append(item["url"])
    for choice in (data.get("output") or {}).get("choices") or []:
        for item in (choice.get("message") or {}).get("content") or []:
            if item.get("image"):
                images.append(item["image"])
    if not images:
        raise RuntimeError(f"接口没有返回图像；request_id={data.get('request_id', '')}")
    return images


def image_bytes(source, timeout=600):
    if source.startswith("data:image/"):
        header, encoded = source.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("图像 Data URL 必须使用 Base64 编码。")
        return base64.b64decode(encoded, validate=True)
    if urlsplit(source).scheme not in ("https", "http"):
        raise ValueError("图像必须为 HTTP(S) URL 或 Base64 Data URL。")
    with requests.get(source, timeout=(30, timeout)) as response:
        response.raise_for_status()
        return response.content
