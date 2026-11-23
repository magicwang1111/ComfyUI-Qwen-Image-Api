import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bailian_api", ROOT / "api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def response(data, status=200, headers=None):
    result = Mock(status_code=status, ok=status < 400, headers=headers or {})
    result.json.return_value = data
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class PayloadTests(unittest.TestCase):
    def test_qwen_openai_flat_t2i(self):
        body = api.qwen_body("qwen-image-3.0-pro", "test", [], size="1024*1536", seed=0)
        self.assertEqual(body["size"], "1024x1536")
        self.assertEqual(body["seed"], 0)
        self.assertEqual(type(body["n"]), int)
        self.assertNotIn("image", body)
        self.assertNotIn("parameters", body)

    def test_qwen_dashscope_image_order(self):
        body = api.qwen_body("qwen-image-2.1-turbo", "edit", ["a", "b"], protocol="dashscope", size="1024x1024")
        self.assertEqual(body["input"]["messages"], [{"role": "user", "content": [{"image": "a"}, {"image": "b"}, {"text": "edit"}]}])
        self.assertEqual(body["parameters"]["size"], "1024*1024")

    def test_qwen_auto_and_random_omitted(self):
        body = api.qwen_body("qwen-image-3.0", "test", [])
        self.assertNotIn("size", body)
        self.assertNotIn("seed", body)

    def test_qwen_capabilities(self):
        for model in api.QWEN_MODELS:
            limit = 3 if "3.0" in model else 10
            with self.subTest(model=model):
                body = api.qwen_body(model, "edit", ["url"] * limit)
                self.assertEqual(len(body["image"]), limit)
                with self.assertRaises(ValueError):
                    api.qwen_body(model, "edit", ["url"] * (limit + 1))
        for kwargs in ({"response_format": "b64_json"}, {"negative_prompt": "blur"}):
            with self.assertRaises(ValueError):
                api.qwen_body("qwen-image-2.1-pro", "test", [], **kwargs)
            api.qwen_body("qwen-image-3.0", "test", [], **kwargs)
        with self.assertRaises(ValueError):
            api.qwen_body("qwen-image-3.0", "edit", ["url"], prompt_extend_mode="agent")

    def test_wan_4k_limits(self):
        for size in ("4K", "4096x4096"):
            api.wan_body("wan2.7-image-pro", "test", [], size=size)
            for model, images, sequential in (("wan2.7-image", [], False), ("wan2.7-image-pro", ["url"], False), ("wan2.7-image-pro", [], True)):
                with self.subTest(size=size, model=model, images=images, sequential=sequential), self.assertRaises(ValueError):
                    api.wan_body(model, "test", images, size=size, enable_sequential=sequential)

    def test_wan_group_count_and_thinking(self):
        body = api.wan_body("wan2.7-image", "test", ["url"], n=12, enable_sequential=True)
        self.assertEqual(body["parameters"]["n"], 12)
        self.assertTrue(body["parameters"]["enable_sequential"])
        self.assertNotIn("thinking_mode", body["parameters"])
        self.assertTrue(api.wan_body("wan2.7-image", "test", [])["parameters"]["thinking_mode"])
        with self.assertRaises(ValueError):
            api.wan_body("wan2.7-image", "test", [], n=5)
        with self.assertRaises(ValueError):
            api.wan_body("wan2.7-image", "test", ["url"] * 10)

    def test_bbox_order_and_validation(self):
        boxes = [[], [[10, 20, 100, 120]]]
        body = api.wan_body("wan2.7-image", "edit", ["a", "b"], bbox_list=json.dumps(boxes))
        self.assertEqual(body["parameters"]["bbox_list"], boxes)
        for bad in ("[]", "{}", "[[[10,20,1,2]],[]]", "[[[0,0,1,2],[0,0,1,2],[0,0,1,2]],[]]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api.wan_body("wan2.7-image", "edit", ["a", "b"], bbox_list=bad)

    def test_color_palette(self):
        palette = json.dumps([{"hex": "#FF0000", "ratio": "33.33%"}, {"hex": "#00FF00", "ratio": "33.33%"}, {"hex": "#0000FF", "ratio": "33.34%"}])
        body = api.wan_body("wan2.7-image", "test", [], color_palette=palette)
        self.assertEqual(body["parameters"]["color_palette"], json.loads(palette))
        with self.assertRaises(ValueError):
            api.wan_body("wan2.7-image", "test", [], color_palette=palette, enable_sequential=True)
        with self.assertRaises(ValueError):
            api.wan_body("wan2.7-image", "test", [], color_palette=palette.replace("33.34", "33.33"))

    def test_invalid_common_parameters(self):
        for kwargs in ({"n": "1"}, {"n": True}, {"seed": 2147483648}, {"size": "1x1"}, {"size": "4096x4096"}, {"size": "8192x128"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.qwen_body("qwen-image-3.0", "test", [], **kwargs)


class TransportTests(unittest.TestCase):
    def test_base_urls_and_explicit_key(self):
        for suffix in ("", "/api/v1", "/compatible-mode/v1", api.OPENAI_PATH, api.SYNC_PATH, api.ASYNC_PATH):
            self.assertEqual(api.connection("https://workspace.example" + suffix, "fake-key"), ("https://workspace.example", "fake-key"))
        self.assertEqual(api.connection("workspace.example", "fake-key")[0], "https://workspace.example")

    @patch.object(api.requests, "request")
    def test_openai_endpoint_headers_and_request_id(self, request):
        request.return_value = response({"data": [{"url": "https://image.example/a.png"}]}, headers={"x-request-id": "rid"})
        result = api.generate({"model": "test"}, "https://workspace.example/api/v1", "fake-key", "openai")
        self.assertEqual(result["request_id"], "rid")
        args, kwargs = request.call_args
        self.assertEqual(args, ("POST", "https://workspace.example" + api.OPENAI_PATH))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake-key")
        self.assertNotIn("X-DashScope-Async", kwargs["headers"])
        self.assertEqual(kwargs["timeout"], (30, 600))

    @patch.object(api.time, "sleep")
    @patch.object(api.requests, "request")
    def test_async_progress_and_result(self, request, sleep):
        request.side_effect = [response({"output": {"task_id": "task-1", "task_status": "PENDING"}}),
                               response({"output": {"task_status": "RUNNING"}}),
                               response({"output": {"task_status": "SUCCEEDED", "choices": [{"message": {"content": [{"image": "https://img/1"}]}}]}})]
        result = api.generate({}, "https://workspace.example", "fake-key", "async")
        self.assertEqual(api.result_images(result), ["https://img/1"])
        self.assertEqual(request.call_args_list[0].args[1], "https://workspace.example" + api.ASYNC_PATH)
        self.assertEqual(request.call_args_list[0].kwargs["headers"]["X-DashScope-Async"], "enable")
        self.assertEqual(request.call_args_list[1].args, ("GET", "https://workspace.example/api/v1/tasks/task-1"))
        self.assertEqual(sum(call.args[0] == "POST" for call in request.call_args_list), 1)

    @patch.object(api.requests, "request")
    def test_resume_never_submits(self, request):
        request.return_value = response({"output": {"task_status": "SUCCEEDED"}})
        api.generate(None, "https://workspace.example", "fake-key", resume_task_id="existing-task")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], "GET")

    @patch.object(api.requests, "request")
    def test_terminal_failures(self, request):
        for status in ("FAILED", "CANCELED", "UNKNOWN"):
            request.return_value = response({"request_id": "rid", "output": {"task_status": status, "message": "failed"}})
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "existing-task.*" + status):
                api.generate(None, "https://workspace.example", "fake-key", resume_task_id="existing-task")

    @patch.object(api.time, "monotonic", side_effect=[0, 601])
    @patch.object(api.requests, "request")
    def test_poll_timeout_has_task_id(self, request, monotonic):
        with self.assertRaisesRegex(TimeoutError, "task-1"):
            api.generate(None, "https://workspace.example", "fake-key", resume_task_id="task-1")
        request.assert_not_called()

    @patch.object(api.requests, "request")
    def test_error_formats_and_key_redaction(self, request):
        for body in ({"error": {"code": "InvalidApiKey", "message": "bad fake-key"}}, {"code": "InvalidApiKey", "message": "bad fake-key"}):
            request.return_value = response(body, 400, {"x-request-id": "rid"})
            with self.assertRaises(RuntimeError) as caught:
                api.request_json("POST", "https://workspace.example", "fake-key", 600)
            self.assertIn("rid", str(caught.exception))
            self.assertNotIn("fake-key", str(caught.exception))

    @patch.object(api.requests, "request", side_effect=api.requests.Timeout())
    def test_no_generation_retry(self, request):
        with self.assertRaisesRegex(RuntimeError, "Timeout"):
            api.generate({}, "https://workspace.example", "fake-key")
        self.assertEqual(request.call_count, 1)

    def test_multiple_choices_base64_and_empty(self):
        self.assertEqual(api.result_images({"data": [{"url": "a"}, {"b64_json": "YWJj"}]}), ["a", "data:image/png;base64,YWJj"])
        self.assertEqual(api.image_bytes("data:image/png;base64,YWJj"), b"abc")
        content = {"output": {"choices": [{"message": {"content": [{"text": "ignored"}, {"image": "a"}]}}, {"message": {"content": [{"image": "b"}]}}]}}
        self.assertEqual(api.result_images(content), ["a", "b"])
        with self.assertRaisesRegex(RuntimeError, "rid"):
            api.result_images({"request_id": "rid", "output": {"choices": None}})


if __name__ == "__main__":
    unittest.main()
