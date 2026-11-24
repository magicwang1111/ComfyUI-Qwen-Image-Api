import json
import unittest
from pathlib import Path


WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows/goumei'


class WorkflowTests(unittest.TestCase):
    def test_1124_uses_optional_details_and_dynamic_reference_count(self):
        graph = json.loads((WORKFLOWS/'20261124qwen全流程（去字幕api）.json').read_text(encoding='utf-8'))
        nodes = {node['id']: node for node in graph['nodes']}
        links = {link[0]: link for link in graph['links']}
        selector = next(node for node in nodes.values() if node['type'] == 'GoumeiSelectLogoDetails')
        pack = next(node for node in nodes.values() if node['type'] == 'GoumeiReferencePack')
        self.assertEqual(selector['widgets_values_named']['selection_mode'], '保留LOGO/文字/花纹')
        self.assertEqual(pack['widgets_values_named']['expected_count'], 0)
        details = next(item for item in pack['inputs'] if item['name'] == 'selected_details')
        self.assertEqual(links[details['link']][1:3], [selector['id'], 4])
        self.assertFalse(any(output['links'] for output in selector['outputs'][:2]))
        self.assertFalse(any(item['link'] for item in pack['inputs'] if item['name'].startswith('extras.')))
        visual = next(node for node in nodes.values() if node['type'] == 'ComfyUI-LLM Qwen Bailian' and '原片 +' in node['title'])
        self.assertIn('只有3张时', visual['widgets_values_named']['system_prompt'])

    def test_master_uses_three_models_five_clothes_and_silent_video(self):
        graph = json.loads((WORKFLOWS/'构美_全流程_规整版.json').read_text(encoding='utf-8'))
        nodes = {node['id']: node for node in graph['nodes']}
        links = {link[0]: link for link in graph['links']}
        by_type = {node['type']: node for node in graph['nodes']}
        def sources(node, prefix):
            return [nodes[links[i['link']][1]] for i in node['inputs'] if i['name'].startswith(prefix) and i['link']]
        front = next(node for node in nodes.values() if node['title'].startswith('A 正面'))
        back = next(node for node in nodes.values() if node['title'].startswith('A 背面'))
        self.assertEqual(len(sources(front, 'reference_images.')), 8)
        self.assertEqual(len(sources(back, 'reference_images.')), 6)
        self.assertEqual(len(sources(by_type['QwenImageAPI'], 'reference_images.')), 3)
        self.assertEqual(len(sources(by_type['GoumeiSelectLogoDetails'], 'clothes.')), 5)
        self.assertEqual(by_type['GoumeiReferencePack']['widgets_values'][2], 5)
        self.assertEqual(sources(by_type['Wan 3.0 API Reference To Video'], 'reference_video_1')[0]['type'], 'Video Preprocess Remove Audio')
        visual = next(node for node in nodes.values() if node['type']=='ComfyUI-LLM Qwen Bailian' and '原片 +' in node['title'])
        self.assertEqual(sources(visual, 'image')[0]['type'], 'GoumeiAnalysisSheet')

    def test_master_automatically_routes_script_and_shared_scenario(self):
        graph = json.loads((WORKFLOWS/'构美_全流程_规整版.json').read_text(encoding='utf-8'))
        nodes = {node['id']: node for node in graph['nodes']}
        links = {link[0]: link for link in graph['links']}
        scenario = next(node for node in nodes.values() if node['type'] == 'GoumeiScenario')
        dialogue = next(node for node in nodes.values() if node['type'] == 'GoumeiAutoDialogue')
        pack = next(node for node in nodes.values() if node['type'] == 'GoumeiReferencePack')
        for node in (dialogue, pack):
            socket = next(item for item in node['inputs'] if item['name'] == 'scenario')
            self.assertEqual(links[socket['link']][1], scenario['id'])
        script = next(item for item in dialogue['inputs'] if item['name'] == 'dialogue_json')
        self.assertEqual(nodes[links[script['link']][1]]['type'], 'ComfyUI-LLM Qwen Bailian')
        self.assertFalse(any(node['type'] == 'GoumeiDialogue' for node in nodes.values()))

    def test_links_have_matching_endpoints_and_no_cycles(self):
        for path in WORKFLOWS.glob('*.json'):
            with self.subTest(file=path.name):
                data = json.loads(path.read_text(encoding='utf-8'))
                nodes = {node['id']: node for node in data['nodes']}
                self.assertEqual(len(nodes), len(data['nodes']))
                incoming = {key: set() for key in nodes}
                for link_id, source, slot, target, input_slot, kind in data['links']:
                    self.assertIn(link_id, nodes[source]['outputs'][slot]['links'])
                    self.assertEqual(nodes[target]['inputs'][input_slot]['link'], link_id)
                    incoming[target].add(source)
                done = set()
                while True:
                    ready = {key for key, dependencies in incoming.items() if key not in done and dependencies <= done}
                    if not ready:
                        break
                    done.update(ready)
                self.assertEqual(done, set(nodes))

    def test_trio_identity_and_back_dependency(self):
        path = WORKFLOWS / '构美_02_角色三图_Wan27_Qwen30.json'
        graph = json.loads(path.read_text(encoding='utf-8'))
        nodes = {node['id']: node for node in graph['nodes']}
        links = {link[0]: link for link in graph['links']}
        back = next(node for node in nodes.values() if node['title'].startswith('A 背面'))
        face = next(node for node in nodes.values() if node['title'].startswith('A 人脸'))
        source = lambda node: nodes[links[next(item['link'] for item in node['inputs'] if item['name']=='reference_images.image0')][1]]
        self.assertTrue(source(back)['title'].startswith('A 正面'))
        self.assertEqual(source(face)['type'], 'LoadImage')
        self.assertEqual(back['widgets_values_named']['model'], 'wan2.7-image-pro')
        self.assertEqual(face['widgets_values_named']['model'], 'qwen-image-3.0-pro')

    def test_final_templates_require_confirmation_and_use_720p(self):
        for path in WORKFLOWS.glob('构美_03*.json'):
            graph = json.loads(path.read_text(encoding='utf-8'))
            by_type = {node['type']: node for node in graph['nodes']}
            self.assertFalse(by_type['GoumeiDialogue']['widgets_values_named']['confirmed'])
            wan = by_type['Wan 3.0 API Reference To Video']
            self.assertEqual(wan['widgets_values'][:3], ['3.0', '', '720P'])
            self.assertEqual(wan['widgets_values'][6], 'Disabled')
            self.assertIsNotNone(next(item['link'] for item in by_type['GoumeiVideoPrompt']['inputs'] if item['name']=='face_hint'))


if __name__ == '__main__':
    unittest.main()
