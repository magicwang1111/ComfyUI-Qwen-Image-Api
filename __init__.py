from comfy_api.latest import ComfyExtension

from .nodes import QwenImageAPI, Wan27ImageAPI


class BailianImageExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImageAPI, Wan27ImageAPI]


async def comfy_entrypoint():
    return BailianImageExtension()
