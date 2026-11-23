from comfy_api.latest import ComfyExtension

from .nodes import QwenImageAPI, Wan27ImageAPI
from .goumei import GoumeiReferencePack, GoumeiDialogue, GoumeiVideoPrompt, GoumeiMuteVideo, GoumeiAnalysisSheet, GoumeiSelectLogoDetails, GoumeiScenario, GoumeiAutoDialogue


class BailianImageExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImageAPI, Wan27ImageAPI, GoumeiReferencePack, GoumeiDialogue, GoumeiVideoPrompt, GoumeiMuteVideo, GoumeiAnalysisSheet, GoumeiSelectLogoDetails, GoumeiScenario, GoumeiAutoDialogue]


async def comfy_entrypoint():
    return BailianImageExtension()
