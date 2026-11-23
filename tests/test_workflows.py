import json
import unittest
from pathlib import Path


WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows/goumei'


class WorkflowTests(unittest.TestCase):
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
