import asyncio
import base64
import importlib.util
import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
spec = importlib.util.spec_from_file_location("bailian_image_plugin", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
nodes = sys.modules["bailian_image_plugin.nodes"]
api = sys.modules["bailian_image_plugin.api"]


def png_url(size=(8, 4), color=(10, 20, 30, 128)):
    with BytesIO() as buffer:
        Image.new("RGBA", size, color).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class NodeTests(unittest.TestCase):
    def test_extension_and_real_comfy_schemas(self):
        extension = asyncio.run(plugin.comfy_entrypoint())
        classes = asyncio.run(extension.get_node_list())
        self.assertEqual(len(classes), 6)
        for cls in classes:
            schema = cls.GET_SCHEMA()
            schema.validate()
            inputs = cls.INPUT_TYPES()
            if cls not in (nodes.QwenImageAPI, nodes.Wan27ImageAPI):
                continue
            self.assertIn("model", inputs["required"])
            self.assertEqual(schema.outputs[0].is_output_list, True)
            ref = next(item for item in schema.inputs if item.id == "reference_images")
            self.assertEqual(ref.template.min, 0)

    def test_different_sizes_and_alpha(self):
        result = nodes.node_result({"data": [{"url": png_url()}, {"url": png_url((3, 2))}], "request_id": "rid"}, 600).result
        images, masks, info = result
        self.assertEqual(images[0].shape, (1, 4, 8, 3))
        self.assertEqual(images[1].shape, (1, 2, 3, 3))
        self.assertEqual(masks[0].shape, (1, 4, 8))
        self.assertAlmostEqual(masks[0][0, 0, 0].item(), 1 - 128 / 255, places=6)
        self.assertEqual(json.loads(info)["image_count"], 2)
        self.assertNotIn("base64", info)

    def test_reference_batch_and_socket_order(self):
        red = torch.zeros((2, 240, 240, 3))
        red[..., 0] = 1
        blue = torch.zeros((1, 240, 240, 3))
        blue[..., 2] = 1
        result = nodes.collect_images({"image2": blue, "image0": red}, "https://example.com/last.png")
        self.assertEqual(len(result), 4)
        for index, color in ((0, (255, 0, 0)), (1, (255, 0, 0)), (2, (0, 0, 255))):
            with Image.open(BytesIO(api.image_bytes(result[index]))) as image:
                self.assertEqual(image.getpixel((0, 0)), color)
        self.assertEqual(result[-1], "https://example.com/last.png")

    def test_wan_alpha_composite_and_minimum_size(self):
        image = torch.zeros((1, 240, 240, 4))
        result = nodes.collect_images({"image0": image}, "", wan=True)
        with Image.open(BytesIO(api.image_bytes(result[0]))) as decoded:
            self.assertEqual(decoded.mode, "RGB")
            self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))
        with self.assertRaises(ValueError):
            nodes.collect_images({"image0": torch.zeros((1, 10, 10, 3))}, "", wan=True)

    @patch.object(api, "generate")
    def test_qwen_node_t2i_i2i(self, generate):
        generate.return_value = {"data": [{"b64_json": png_url().split(",")[1]}]}
        nodes.QwenImageAPI.execute("qwen-image-3.0-pro", "test")
        self.assertNotIn("image", generate.call_args.args[0])
        self.assertEqual(generate.call_args.args[3], "openai")
        nodes.QwenImageAPI.execute("qwen-image-3.0", "edit", reference_images={"image0": torch.zeros((1, 4, 4, 3))})
        self.assertEqual(len(generate.call_args.args[0]["image"]), 1)

    @patch.object(api, "generate")
    def test_wan_node_group_and_resume(self, generate):
        generate.return_value = {"output": {"choices": [{"message": {"content": [{"image": png_url()}]}}]}}
        nodes.Wan27ImageAPI.execute("wan2.7-image", "group", call_mode="async", n=4, enable_sequential=True)
        self.assertEqual(generate.call_args.args[3], "async")
        self.assertTrue(generate.call_args.args[0]["parameters"]["enable_sequential"])
        nodes.Wan27ImageAPI.execute("wan2.7-image", "", resume_task_id="existing")
        self.assertIsNone(generate.call_args.args[0])
        self.assertEqual(generate.call_args.args[-1], "existing")


if __name__ == "__main__":
    unittest.main()
