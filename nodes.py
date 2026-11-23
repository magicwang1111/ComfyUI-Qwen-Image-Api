import base64
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps
from comfy_api.latest import io
from comfy.model_management import throw_exception_if_processing_interrupted

from . import api


def reference_inputs(maximum):
    return [
        io.String.Input("image_urls", default="", multiline=True, optional=True,
                        tooltip="每行一个公网 URL 或 Base64 Data URL，排在已连接参考图之后。"),
        io.Autogrow.Input("reference_images", template=io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True), prefix="image", min=0, max=maximum,
        ), optional=True, tooltip="不接图为文生图；按插口编号、batch 内顺序传入全部参考图。"),
    ]


def connection_inputs():
    return [
        io.String.Input("base_url", default="", optional=True, advanced=True,
                        tooltip="留空读取本地配置。支持 Host、DashScope 或 OpenAI base_url。"),
        io.String.Input("api_key", default="", optional=True, advanced=True,
                        tooltip="建议留空读取环境变量或 config.local.json；直接填写会随工作流保存。"),
        io.Int.Input("timeout", default=600, min=30, max=86400, optional=True, advanced=True,
                     tooltip="同步请求超时或异步轮询等待预算（秒）。"),
    ]


def output_schema():
    return [io.Image.Output("images", is_output_list=True), io.Mask.Output("masks", is_output_list=True),
            io.String.Output("info")]


def collect_images(reference_images, image_urls, wan=False):
    sources = []
    for name in sorted(reference_images or {}, key=lambda name: int(name.removeprefix("image"))):
        batch = reference_images[name]
        if batch is None:
            continue
        for tensor in batch:
            throw_exception_if_processing_interrupted()
            array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(array)
            # Wan does not accept alpha; composite transparent inputs onto white.
            if wan and image.mode == "RGBA":
                background = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(background, image).convert("RGB")
            minimum, ratio, max_mb = (240, 8, 20) if wan else (1, 10, 10)
            width, height = image.size
            if min(width, height) < minimum or max(width, height) > 8000 or max(width, height) > ratio * min(width, height):
                raise ValueError(f"参考图宽高须为 {minimum}–8000 像素，长短边比不超过 {ratio}:1；当前 {width}×{height}。")
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                raw = buffer.getvalue()
            if len(raw) > max_mb * 1024 * 1024:
                raise ValueError(f"参考图编码后超过 {max_mb} MB，请缩小图片后重试。")
            sources.append("data:image/png;base64," + base64.b64encode(raw).decode("ascii"))
    for source in image_urls.splitlines():
        source = source.strip()
        if not source:
            continue
        if not source.startswith(("https://", "http://", "data:image/")):
            raise ValueError("image_urls 每行须为 HTTP(S) URL 或 Base64 Data URL。本地图片请连接 Load Image。")
        sources.append(source)
    return sources


def node_result(data, timeout):
    images, masks = [], []
    for source in api.result_images(data):
        throw_exception_if_processing_interrupted()
        with Image.open(BytesIO(api.image_bytes(source, timeout))) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array[..., :3].copy()).unsqueeze(0))
        masks.append(torch.from_numpy(1.0 - array[..., 3].copy()).unsqueeze(0))
    output = data.get("output") or {}
    info = {"request_id": data.get("request_id", ""), "usage": data.get("usage", {}),
            "image_count": len(images)}
    for key in ("task_id", "task_status", "rewrite_status"):
        if key in output:
            info[key] = output[key]
    return io.NodeOutput(images, masks, json.dumps(info, ensure_ascii=False, indent=2))


class QwenImageAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImageAPI", display_name="千问图像 API / Qwen Image", category="image/百炼 API",
            description="千问文生图与多参考图编辑。默认 OpenAI 兼容接口。",
            inputs=[
                io.Combo.Input("model", options=api.QWEN_MODELS),
                io.String.Input("prompt", default="", multiline=True),
                io.Combo.Input("protocol", options=["openai", "dashscope"], default="openai"),
                io.String.Input("size", default="auto", tooltip="auto 自动推荐，或宽x高，如 1024x1024。"),
                io.Int.Input("n", default=1, min=1, max=6),
                io.Int.Input("seed", default=0, min=-1, max=2147483647, control_after_generate=True,
                             tooltip="-1 不传种子，由服务随机选择。"),
                io.Boolean.Input("prompt_extend", default=True),
                io.Combo.Input("prompt_extend_mode", options=["direct", "agent"], default="direct"),
                io.Boolean.Input("enable_thinking", default=True),
                io.Boolean.Input("watermark", default=False),
                io.String.Input("negative_prompt", default="", multiline=True, optional=True,
                                tooltip="仅千问 3.0 系列支持。"),
                io.Combo.Input("response_format", options=["url", "b64_json"], default="url", optional=True,
                               tooltip="b64_json 仅千问 3.0 系列支持。"),
                *reference_inputs(10), *connection_inputs(),
            ], outputs=output_schema(),
        )

    @classmethod
    def fingerprint_inputs(cls, seed=0, **kwargs):
        return float("nan") if seed == -1 else seed

    @classmethod
    def execute(cls, model, prompt, protocol="openai", size="auto", n=1, seed=0,
                prompt_extend=True, prompt_extend_mode="direct", enable_thinking=True, watermark=False,
                negative_prompt="", response_format="url", image_urls="", reference_images=None,
                base_url="", api_key="", timeout=600):
        images = collect_images(reference_images, image_urls)
        body = api.qwen_body(model, prompt, images, protocol, size, n, seed, prompt_extend,
                             prompt_extend_mode, enable_thinking, watermark, negative_prompt, response_format)
        data = api.generate(body, base_url, api_key, protocol, timeout, throw_exception_if_processing_interrupted)
        return node_result(data, timeout)


class Wan27ImageAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Wan27ImageAPI", display_name="万相 2.7 图像 API / Wan", category="image/百炼 API",
            description="万相文生图、图像编辑、参考图生成与组图。DashScope 原生同步/异步调用。",
            inputs=[
                io.Combo.Input("model", options=api.WAN_MODELS),
                io.String.Input("prompt", default="", multiline=True),
                io.Combo.Input("call_mode", options=["sync", "async"], default="sync"),
                io.String.Input("size", default="2K", tooltip="1K、2K、4K 或宽*高。4K 仅 Pro 非组图文生图。"),
                io.Boolean.Input("enable_sequential", default=False),
                io.Int.Input("n", default=1, min=1, max=12, tooltip="普通生成 1–4 张；组图 1–12 张（最大数量）。"),
                io.Int.Input("seed", default=0, min=-1, max=2147483647, control_after_generate=True),
                io.Boolean.Input("thinking_mode", default=True),
                io.Boolean.Input("watermark", default=False),
                io.String.Input("bbox_list", default="", multiline=True, optional=True,
                                tooltip="框选 JSON，如 [[], [[989,515,1138,681]]]；与参考图逐张对应。"),
                io.String.Input("color_palette", default="", multiline=True, optional=True,
                                tooltip='非组图可用，3–10 色，比例和 100.00%。如 [{"hex":"#FF0000","ratio":"50.00%"}, ...]'),
                io.String.Input("resume_task_id", default="", optional=True, advanced=True,
                                tooltip="填写已有任务 ID 时仅查询结果，忽略生成参数，不重新提交。"),
                *reference_inputs(9), *connection_inputs(),
            ], outputs=output_schema(),
        )

    @classmethod
    def fingerprint_inputs(cls, seed=0, resume_task_id="", **kwargs):
        return float("nan") if seed == -1 and not resume_task_id.strip() else seed

    @classmethod
    def execute(cls, model, prompt, call_mode="sync", size="2K", enable_sequential=False, n=1, seed=0,
                thinking_mode=True, watermark=False, bbox_list="", color_palette="", resume_task_id="",
                image_urls="", reference_images=None, base_url="", api_key="", timeout=600):
        body = None
        if not resume_task_id.strip():
            images = collect_images(reference_images, image_urls, wan=True)
            body = api.wan_body(model, prompt, images, size, n, seed, enable_sequential,
                                thinking_mode, watermark, bbox_list, color_palette)
        data = api.generate(body, base_url, api_key, "async" if call_mode == "async" else "dashscope",
                            timeout, throw_exception_if_processing_interrupted, resume_task_id)
        return node_result(data, timeout)
