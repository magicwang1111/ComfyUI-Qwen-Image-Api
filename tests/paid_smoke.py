"""Explicit paid integration test: four requests, one image per request."""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
spec = importlib.util.spec_from_file_location("bailian_paid_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
nodes = sys.modules["bailian_paid_test.nodes"]


def main():
    destination = COMFY_ROOT / "output" / ("qwen_wan_api_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True)
    report = []

    def run(name, node, **kwargs):
        started = time.monotonic()
        print(f"START {name}", flush=True)
        try:
            images, masks, info = node.execute(**kwargs).result
            paths = []
            for index, (image, mask) in enumerate(zip(images, masks)):
                rgb = (image[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                alpha = ((1 - mask[0].numpy()) * 255).clip(0, 255).astype(np.uint8)
                path = destination / f"{name}_{index + 1}.png"
                Image.fromarray(np.dstack((rgb, alpha))).save(path)
                paths.append(str(path))
            report.append({"test": name, "status": "passed", "seconds": round(time.monotonic() - started, 2),
                           "files": paths, "info": json.loads(info)})
            print(f"PASS {name}: {len(images)} image(s), {report[-1]['seconds']}s", flush=True)
            return images[0]
        except Exception as exc:
            report.append({"test": name, "status": "failed", "error": str(exc), "seconds": round(time.monotonic() - started, 2)})
            print(f"FAIL {name}: {exc}", flush=True)
            return None
        finally:
            (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt = "简洁的产品摄影：一只黄色陶瓷花瓶，插着三朵白色雏菊，纯浅灰背景，柔和自然光，正方形构图。"
    qwen = run("qwen_t2i", nodes.QwenImageAPI, model="qwen-image-3.0", prompt=prompt,
               size="1024x1024", n=1, seed=2026, response_format="b64_json", timeout=600)
    if qwen is not None:
        run("qwen_i2i", nodes.QwenImageAPI, model="qwen-image-3.0",
            prompt="将黄色花瓶改成蓝色，保留三朵白色雏菊、构图和背景。", reference_images={"image0": qwen},
            size="1024x1024", n=1, seed=2026, response_format="url", timeout=600)
    wan = run("wan_async_t2i", nodes.Wan27ImageAPI, model="wan2.7-image", prompt=prompt,
              call_mode="async", size="1K", n=1, seed=2026, timeout=600)
    if wan is not None:
        run("wan_sync_i2i", nodes.Wan27ImageAPI, model="wan2.7-image",
            prompt="将黄色花瓶改成蓝色，保留三朵白色雏菊、构图和背景。", reference_images={"image0": wan},
            call_mode="sync", size="1K", n=1, seed=2026, timeout=600)
    print(f"REPORT {destination / 'report.json'}", flush=True)
    return 0 if len(report) == 4 and all(item["status"] == "passed" for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
