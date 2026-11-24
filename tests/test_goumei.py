import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import av
import torch

from test_nodes import plugin
from bailian_image_plugin.goumei import GoumeiReferencePack, GoumeiDialogue, GoumeiVideoPrompt, GoumeiMuteVideo, GoumeiAnalysisSheet, GoumeiSelectLogoDetails, GoumeiScenario, GoumeiAutoDialogue, VideoFromFile, dialogue_timeline


class GoumeiTests(unittest.TestCase):
    def test_automatic_dialogue_routes_without_confirmation(self):
        voice = {"name": "voiceover"}
        output = GoumeiAutoDialogue.execute("单人＋画外音", '{"segments":[{"role":"B","text":"看看这件绿色外套"}]}',
            '{"duration":5.2}', voice_b=voice).result
        self.assertEqual(output[1], 6)
        self.assertIs(output[2], voice)
        self.assertIsNone(output[3])
        self.assertEqual(json.loads(output[4])["mode"], "automatic")
        self.assertIn("自动编排台词", output[0])
        self.assertNotIn("confirmed", {item.id for item in GoumeiAutoDialogue.define_schema().inputs})

    def test_automatic_dialogue_rejects_invalid_roles_and_missing_voice(self):
        for scenario, script, voices in [("单人口播", '[{"role":"B","text":"你好"}]', {"voice_b": {}}),
                                          ("单人口播", '[{"role":"A","text":"你好"}]', {})]:
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                GoumeiAutoDialogue.execute(scenario, script, '{"duration":5}', **voices)

    def test_scenario_has_one_control_and_excludes_dual(self):
        self.assertEqual(len(GoumeiScenario.define_schema().inputs), 1)
        self.assertIn("所有台词role为A", GoumeiScenario.execute("单人口播").result[1])
        with self.assertRaises(ValueError):
            GoumeiScenario.execute("双人口播")

    def test_select_logo_returns_original_tensors_in_rank_order(self):
        originals = {f'image{i}': torch.full((1, 8+i, 8, 3), i/5) for i in range(5)}
        report = {"images": [{"index": i, "logo_text_closeup": i in (2, 5), "readable": True,
                              "matches_selected_item": True, "confidence": 0.99 if i == 5 else 0.9,
                              "description": f"detail {i}"} for i in range(1,6)]}
        first, second, labels, selection = GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals).result[:4]
        self.assertIs(first, originals['image4'])
        self.assertIs(second, originals['image1'])
        self.assertEqual(json.loads(selection)['selected_original_indices'], [5, 2])
        self.assertEqual(len(json.loads(labels)), 2)

    def test_logo_insufficient_never_duplicates_or_fills(self):
        originals = {f'image{i}': torch.zeros((1, 4, 4, 3)) for i in range(5)}
        report = {"images": [{"index": i, "logo_text_closeup": i == 2, "readable": True,
                              "matches_selected_item": True, "confidence": 0.99} for i in range(1,6)]}
        with self.assertRaisesRegex(ValueError, "只识别到 1 张"):
            GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals)
        report['images'][4].update(logo_text_closeup=True, readable=False)
        with self.assertRaises(ValueError):
            GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals)

    def test_logo_manual_indices_are_distinct_and_in_range(self):
        originals = {f'image{i}': torch.zeros((1, 4, 4, 3)) for i in range(5)}
        output = GoumeiSelectLogoDetails.execute('', '[2,5]', originals).result
        self.assertIs(output[0], originals['image1'])
        for value in ('[2,2]', '[0,5]', '[2,6]', '[true,5]', '[2]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GoumeiSelectLogoDetails.execute('', value, originals)

    def test_analysis_sheet_is_single_numbered_image(self):
        frames = torch.zeros((5, 10, 5, 3))
        result = GoumeiAnalysisSheet.execute({'image0': frames}).result[0]
        self.assertEqual(result.shape, (1, 2160, 3072, 3))

    def test_optional_details_keep_logo_text_or_pattern_without_old_gates(self):
        originals = {f'image{i}': torch.full((1, 4+i, 4, 3), i/5) for i in range(5)}
        report = {"images": [{"index": i, "has_logo": i == 1, "has_text": i == 3, "has_pattern": i == 5,
                              "logo_text_closeup": False, "readable": False, "matches_selected_item": False,
                              "confidence": 0.1, "description": f"detail {i}"} for i in range(1, 6)]}
        result = GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals, selection_mode="保留LOGO/文字/花纹").result
        self.assertEqual(json.loads(result[3])['selected_original_indices'], [1, 3, 5])
        for image, index in zip(result[4]['images'], [0, 2, 4]):
            self.assertIs(image, originals[f'image{index}'])

    def test_optional_details_zero_to_five_reach_reference_pack_and_prompt(self):
        originals = {f'image{i}': torch.full((1, 4+i, 4, 3), i/5) for i in range(5)}
        image = originals['image0']
        for count in range(6):
            with self.subTest(count=count):
                report = {"images": [{"index": i, "has_logo": False, "has_text": False, "has_pattern": i <= count,
                                      "description": f"pattern {i}"} for i in range(1, 6)]}
                selected = GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals, selection_mode="保留LOGO/文字/花纹").result
                self.assertEqual(len(selected[4]['images']), count)
                if count == 0:
                    self.assertIsNone(selected[0])
                    self.assertIsNone(selected[1])
                    self.assertEqual(json.loads(selected[2]), [])
                packed, manifest = GoumeiReferencePack.execute('单人＋画外音', image, image, image, selected_details=selected[4]).result
                self.assertEqual(packed.shape[0], 3 + count)
                entries = json.loads(manifest)['images']
                self.assertEqual([item['number'] for item in entries], list(range(1, 4 + count)))
                self.assertEqual([item['purpose'] for item in entries[3:]], selected[4]['labels'])
                prompt = GoumeiVideoPrompt.execute(manifest, '', '台词', '{"scenario":"单人＋画外音"}').result[0]
                self.assertNotIn(f"图片{4+count}：", prompt)
                self.assertIn('没有原始细节图时仅依据正面、背面和人脸参考', prompt)

    def test_optional_details_manual_empty_single_and_all(self):
        originals = {f'image{i}': torch.zeros((1, 4, 4, 3)) for i in range(5)}
        for chosen in ([], [3], [1, 2, 3, 4, 5]):
            result = GoumeiSelectLogoDetails.execute('', json.dumps(chosen), originals, '保留LOGO/文字/花纹').result
            self.assertEqual(json.loads(result[3])['selected_original_indices'], chosen)

    def test_optional_details_invalid_analysis_is_not_treated_as_no_details(self):
        originals = {f'image{i}': torch.zeros((1, 4, 4, 3)) for i in range(5)}
        report = {"images": [{"index": i, "has_logo": False, "has_text": False, "has_pattern": "false"} for i in range(1, 6)]}
        with self.assertRaisesRegex(ValueError, "true/false"):
            GoumeiSelectLogoDetails.execute(json.dumps(report), clothes=originals, selection_mode='保留LOGO/文字/花纹')

    def test_exact_five_reference_pack(self):
        image = torch.zeros((1, 4, 4, 3))
        with self.assertRaisesRegex(ValueError, "必须输入 5"):
            GoumeiReferencePack.execute('单人＋画外音', image, image, image, expected_count=5)
        packed = GoumeiReferencePack.execute('单人＋画外音', image, image, image, '["原图2 LOGO","原图5文字"]',
                    extras={'image0': image, 'image1': image}, expected_count=5).result
        self.assertEqual(packed[0].shape[0], 5)
        self.assertIn('原图2', json.loads(packed[1])['images'][3]['purpose'])

    def test_timing_merge_and_continuity(self):
        result = dialogue_timeline('[{"role":"B","text":"你好"},{"role":"A","text":"欢迎"},{"role":"A","text":"请看这件衣服"}]', 14, "单人＋画外音", "按字数连续分配")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[0]["end"], result[1]["start"])
        self.assertEqual(result[-1]["end"], 14)
        self.assertEqual(result[0]["role"], "B")

    def test_voiceover_only_remaps_audio(self):
        voice_a, voice_b = {"name": "A"}, {"name": "B"}
        result = GoumeiDialogue.execute("单人＋画外音", '[{"role":"B","text":"介绍商品"}]', '{"duration":13.7}', "按字数连续分配", True, voice_a, voice_b).result
        self.assertEqual(result[1], 14)
        self.assertIs(result[2], voice_b)
        self.assertIsNone(result[3])
        self.assertEqual(json.loads(result[4])["audio_map"], {"B": 1})
        self.assertIn("A 不发声", result[0])

    def test_dialogue_requires_confirmation_and_voices(self):
        for confirmed, voice in ((False, {}), (True, None)):
            with self.assertRaises(ValueError):
                GoumeiDialogue.execute("单人口播", '[{"role":"A","text":"测试"}]', '{"duration":5}', "按字数连续分配", confirmed, voice)

    def test_manual_times_reject_overlap(self):
        with self.assertRaises(ValueError):
            dialogue_timeline('[{"role":"A","text":"你好","start":0,"end":3},{"role":"B","text":"好","start":2,"end":4}]', 5, "双人口播", "保留确认时间")

    def test_single_does_not_accept_unknown_or_b(self):
        for role in ("B", "speaker_0"):
            with self.assertRaises(ValueError):
                dialogue_timeline(json.dumps([{"role": role, "text": "你好"}]), 5, "单人口播", "按字数连续分配")

    def test_pack_order_and_aspect_ratio(self):
        front = torch.zeros((1, 80, 40, 3))
        back = torch.ones((1, 60, 30, 3))
        face = torch.ones((1, 40, 40, 3)) * 0.5
        images, manifest = GoumeiReferencePack.execute("单人＋画外音", front, back, face).result
        self.assertEqual(images.shape, (3, 1536, 1536, 3))
        self.assertEqual(images[0, 500, 100].tolist(), [1, 1, 1])
        self.assertEqual(images[0, 500, 700].tolist(), [0, 0, 0])
        self.assertIn("人脸", json.loads(manifest)["images"][2]["purpose"])

    def test_pack_dual_and_limit_and_extra_labels(self):
        image = torch.zeros((1, 4, 4, 3))
        with self.assertRaises(ValueError):
            GoumeiReferencePack.execute("双人口播", image, image, image)
        with self.assertRaises(ValueError):
            GoumeiReferencePack.execute("单人口播", image, image, image, extras={"image0": image})
        extras = torch.zeros((8, 4, 4, 3))
        with self.assertRaisesRegex(ValueError, "最多 10"):
            GoumeiReferencePack.execute("单人口播", image, image, image, json.dumps(["logo"] * 8), extras={"image0": extras})

    def test_scenario_mismatch(self):
        with self.assertRaises(ValueError):
            GoumeiVideoPrompt.execute('{"scenario":"单人口播","images":[]}', "", "", '{"scenario":"双人口播"}')

    def test_prompt_preserves_dialogue_and_mask_hint(self):
        prompt = GoumeiVideoPrompt.execute('{"scenario":"单人口播","images":[{"number":3,"purpose":"A人脸"}]}', "白色外套", "逐字台词", '{"scenario":"单人口播"}', face_hint="必须还原图3人脸").result[0]
        self.assertIn("逐字台词", prompt)
        self.assertIn("图片3", prompt)
        self.assertIn("必须还原图3人脸", prompt)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_reference_mute_preserves_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=256x256:r=24:d=2',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(source)], check=True)
            result = GoumeiMuteVideo.execute(VideoFromFile(str(source))).result[0]
            destination = Path(directory) / 'muted.mp4'
            result.save_to(str(destination))
            with av.open(str(destination)) as video:
                self.assertEqual(len(video.streams.audio), 0)
                self.assertEqual(len(video.streams.video), 1)
                self.assertEqual(video.streams.video[0].width, 256)
                self.assertAlmostEqual(video.duration / av.time_base, 2, places=1)


if __name__ == "__main__":
    unittest.main()
