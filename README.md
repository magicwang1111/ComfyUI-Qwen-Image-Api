# ComfyUI Qwen / Wan Image API

阿里云百炼图像生成与编辑节点，使用当前 ComfyUI V3 API（需支持 `io.Autogrow`）。

| 节点 | 模型 | 接口 |
| --- | --- | --- |
| 千问图像 API / Qwen Image | qwen-image-3.0-pro、qwen-image-3.0、qwen-image-2.1-pro、qwen-image-2.1-turbo | 默认 OpenAI 兼容，同步；可切换 DashScope 同步 |
| 万相 2.7 图像 API / Wan | wan2.7-image-pro、wan2.7-image | DashScope 原生，同步或异步轮询 |

## 安装与配置

放入 `ComfyUI/custom_nodes/ComfyUI-Qwen-Image-Api`，使用 ComfyUI 的 Python 安装 `requirements.txt`（已有依赖无需重复安装），重启 ComfyUI。在节点搜索中输入 `QwenImageAPI` / `Wan27ImageAPI`，或在 `image/百炼 API` 分类下添加。

复制 `config.example.json` 为 `config.local.json`，填写业务空间域名及 API Key。该本地文件已加入 `.gitignore`，不要分享。节点的 `base_url` 与 `api_key` 留空即可读取配置。

配置优先级：节点非空输入 → `DASHSCOPE_BASE_URL` / `DASHSCOPE_API_KEY` 环境变量 → 本地配置。没有域名配置时使用北京旧域名 `https://dashscope.aliyuncs.com`。模型、域名与 Key 必须属于同一地域。

支持填写 Host（可省略 `https://`）、DashScope `/api/v1`、OpenAI `/compatible-mode/v1` 或完整生成接口地址。节点根据协议选择正确端点：

- Qwen OpenAI：`/compatible-mode/v1/images/generations`，编辑图通过 JSON `image` 传入。
- Qwen / Wan DashScope 同步：`/api/v1/services/aigc/multimodal-generation/generation`。
- Wan 异步提交：`/api/v1/services/aigc/image-generation/generation`，带 `X-DashScope-Async: enable`。
- Wan 异步查询：`/api/v1/tasks/{task_id}`。

## 使用

1. **文生图**：填写 `prompt`，不连接参考图；将 `images` 输出连接 `Save Image` 或 `Preview Image`。
2. **图像编辑 / 多图参考**：把 `Load Image` 连接到动态参考图插口，继续连接即可添加更多插口。参考顺序为插口编号顺序、各 batch 内顺序，最后追加 `image_urls` 中逐行填写的 URL / Base64 Data URL。所有图片均作为一次调用的参考图。
3. **万相组图**：开启 `enable_sequential`，设置 `n`（1–12，表示最多生成几张），提示词描述各张内容；可连接参考图。
4. **万相框选编辑**：在 `bbox_list` 填 JSON，例如两张输入图、仅框选第二张：`[[], [[989,515,1138,681]]]`。坐标为原图像素，每张最多 2 个框。
5. **万相配色**：关闭组图后填写 `color_palette`，例如 `[{"hex":"#FF0000","ratio":"50.00%"},{"hex":"#00FF00","ratio":"25.00%"},{"hex":"#0000FF","ratio":"25.00%"}]`。

输出 `images` 和 `masks` 为一一对应的 ComfyUI 列表，保留各张原始分辨率，连接保存节点会逐张保存。`images` 为 RGB，`masks` 使用 ComfyUI 约定（1=透明，0=不透明）；需要保存透明 PNG 时，把两者接到 `Join Image with Alpha` 再保存。`info` 输出请求 ID、用量、实际图片数以及异步任务 ID。

### 参数范围

| 参数 | Qwen | Wan 2.7 |
| --- | --- | --- |
| 参考图 | 3.0 最多 3 张；2.1 最多 10 张 | 最多 9 张 |
| 输出数量 n | 1–6 | 普通 1–4；组图最多 1–12 |
| size | `auto` 或宽高；面积 512²–2048² | `1K`、`2K` 或宽高；面积 768²–2048²；Pro 无图非组图支持 `4K` / 最大 4096² |
| 宽高比 | 1:8–8:1 | 1:8–8:1 |
| seed | 0–2147483647；-1 由服务随机 | 同左 |
| 负面提示词 | 仅 3.0 | 不提供 |
| Base64 返回 | 仅 3.0 同步 | URL 返回 |

`size` 接受 `1024x1024` 或 `1024*1024`，自动转换协议分隔符。Qwen `auto` 不发送 size，交由服务推荐。Wan `auto` 不发送 size，使用服务默认（2K）。万相 K 规格有参考图时跟随最后一张图的比例；无参考图时为正方形。

Qwen `agent` 改写仅用于文生图，编辑使用 `direct`。`enable_thinking` 仅在开启 `prompt_extend` 时生效。Wan `thinking_mode` 仅在无参考图且关闭组图时发送。

本地参考图以 PNG Base64 发送，不自动缩放。Qwen 宽高 1–8000、长短边比不超过 10:1、编码后不超过 10 MB；Wan 宽高 240–8000、比例不超过 8:1、编码后不超过 20 MB。Wan 的 RGBA 输入合成到白底 RGB。公网 URL 和手动填写的 Data URL 由服务检查图片规格，请勿向 Wan 传透明 PNG。

### 异步与超时

Wan `call_mode=async` 提交一次任务并轮询至完成。默认超时 600 秒，异步中该值为提交之后的等待预算；同步中为 HTTP 读取超时。API 失败不会自动重新提交生成请求。

终端会显示 `task_id`。等待超时或查询连接失败后，将该 ID 填入 `resume_task_id` 可继续取图；此时不提交新任务，生成参数被忽略。点击 ComfyUI 停止会在轮询等待期间响应；正在执行的同步 HTTP 请求需等响应或超时。停止本地等待不会取消云端任务。链接与任务结果有效期约 24 小时，请及时连接保存节点。

固定种子的重复运行遵循 ComfyUI 缓存；使用 seed 的 `randomize` 或 `-1` 生成新结果。API Key 直接填写到节点会随工作流保存，因此建议使用本地配置或环境变量。

## 验证

在插件目录用 ComfyUI Python 运行：

```shell
python -m unittest discover -s tests -v
```

测试覆盖请求与端点、模型参数限制、异步成功/失败/超时与续查、节点注册、多图顺序、不同尺寸输出和透明蒙版。测试使用模拟响应，不调用付费生图接口。无需安装 OpenAI 或 DashScope SDK。

2026-11-23 实测通过：Qwen 3.0 OpenAI 文生图（Base64 返回）及参考图编辑（URL 返回）、Wan 2.7 原生异步文生图及同步编辑，共 4 张 1024×1024 图片。两种模型均完成黄色花瓶改成蓝色的编辑。

手动运行 `python tests/paid_smoke.py` 可重复上述 **4 次付费调用**，图片和请求记录保存到 `ComfyUI/output/qwen_wan_api_test_时间戳/`。该脚本不会被 unittest 自动执行。
