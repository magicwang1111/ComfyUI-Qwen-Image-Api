import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from comfy_api.latest import io
from comfy_api.latest._input_impl import VideoFromFile


SCENARIOS = ["单人口播", "双人口播", "单人＋画外音"]


def json_value(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


class GoumeiAnalysisSheet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiAnalysisSheet", display_name="构美｜带编号分析预览", category="image/百炼 API/构美",
            description="仅给视觉分析或预览使用。原始独立图片仍直接传给生成/选择节点。",
            inputs=[io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                io.Image.Input("image"), prefix="image", min=1, max=10))], outputs=[io.Image.Output("sheet")])

    @classmethod
    def execute(cls, images):
        frames = [frame for name in sorted(images, key=lambda name: int(name.removeprefix("image"))) for frame in images[name]]
        if not 1 <= len(frames) <= 10:
            raise ValueError("分析预览须包含 1–10 张图。")
        columns = min(3, len(frames))
        canvas = Image.new("RGB", (columns * 1024, math.ceil(len(frames) / columns) * 1080), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=36)
        for index, frame in enumerate(frames):
            image = Image.fromarray((frame.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
            image = ImageOps.contain(image, (1024, 1024), Image.Resampling.LANCZOS)
            x, y = (index % columns) * 1024, (index // columns) * 1080
            canvas.paste(image, (x + (1024 - image.width) // 2, y + 56 + (1024 - image.height) // 2))
            draw.text((x + 20, y + 8), f"IMAGE {index + 1}", font=font, fill="black")
        return io.NodeOutput(torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0))


class GoumeiSelectLogoDetails(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiSelectLogoDetails", display_name="构美｜从五张原图选择两张LOGO细节", category="image/百炼 API/构美",
            inputs=[io.String.Input("analysis_json", force_input=True),
                    io.String.Input("manual_indices", default="", tooltip="留空自动选择。可手填原图编号，如 [2,5]；保留LOGO/文字/花纹模式允许0–5张，[] 表示不保留。"),
                    io.Autogrow.Input("clothes", template=io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=5, max=5)),
                    io.Combo.Input("selection_mode", options=["两张LOGO特写", "保留LOGO/文字/花纹"], default="两张LOGO特写", optional=True)],
            outputs=[io.Image.Output("detail_1"), io.Image.Output("detail_2"), io.String.Output("labels"), io.String.Output("selection_report"),
                     io.Custom("GOUMEI_DETAILS").Output("selected_details")])

    @classmethod
    def execute(cls, analysis_json, manual_indices="", clothes=None, selection_mode="两张LOGO特写"):
        originals = [clothes[name] for name in sorted(clothes or {}, key=lambda name: int(name.removeprefix("image")))]
        if len(originals) != 5 or any(len(image) != 1 for image in originals):
            raise ValueError("须连接五张独立的原始服饰图，每个插口仅含一张。")
        descriptions = {}
        flexible = selection_mode == "保留LOGO/文字/花纹"
        manual = bool(manual_indices.strip())
        if manual:
            chosen = json_value(manual_indices)
        else:
            analysis = json_value(analysis_json)
            items = analysis.get("images") if isinstance(analysis, dict) else None
            if not isinstance(items, list) or len(items) != 5:
                raise ValueError("分析结果必须包含 images 数组，逐张描述原始图片1–5。")
            indexed = {}
            for item in items:
                if not isinstance(item, dict) or type(item.get("index")) is not int or not 1 <= item["index"] <= 5 or item["index"] in indexed:
                    raise ValueError("服饰图分析编号必须恰好为1–5，不允许重复或越界。")
                if flexible:
                    if any(type(item.get(field)) is not bool for field in ("has_logo", "has_text", "has_pattern")):
                        raise ValueError("每张图须用 true/false 标明 has_logo、has_text、has_pattern。")
                else:
                    score = item.get("confidence")
                    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("每张图的 confidence 必须为0–1的数字。")
                indexed[item["index"]] = item
            if flexible:
                candidates = [indexed[index] for index in sorted(indexed)
                              if any(indexed[index][field] for field in ("has_logo", "has_text", "has_pattern"))]
            else:
                candidates = [item for item in indexed.values() if item.get("logo_text_closeup") is True
                              and item.get("readable") is True and item.get("matches_selected_item") is True and item["confidence"] >= 0.8]
                if len(candidates) < 2:
                    raise ValueError(f"只识别到 {len(candidates)} 张清晰且匹配目标单品的LOGO/文字特写，不能满足两张原始细节图。请更换素材或目视确认后填写 manual_indices；不会重复图片或生成LOGO凑数。")
                candidates.sort(key=lambda item: (-item["confidence"], item["index"]))
                candidates = candidates[:2]
            chosen = [item["index"] for item in candidates]
            descriptions = {item["index"]: str(item.get("description", "LOGO/文字/花纹细节")) for item in candidates}
        if (not isinstance(chosen, list) or (not flexible and len(chosen) != 2)
                or any(type(index) is not int or not 1 <= index <= 5 for index in chosen) or len(set(chosen)) != len(chosen)):
            raise ValueError("原图编号须为1–5且不能重复；两张LOGO特写模式须选两个编号。")
        labels = [f"A 原始服饰图{index}：{descriptions.get(index, '人工确认的LOGO/文字细节')}" for index in chosen]
        report = {"selected_original_indices": chosen, "selection": "manual" if manual else "model_analysis",
                  "labels": labels, "source": "original_clothing_images", "generated_or_cropped": False}
        selected = [originals[index - 1] for index in chosen]
        return io.NodeOutput(selected[0] if selected else None, selected[1] if len(selected) > 1 else None,
                             json.dumps(labels, ensure_ascii=False), json.dumps(report, ensure_ascii=False, indent=2),
                             {"images": selected, "labels": labels})


class GoumeiReferencePack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiReferencePack", display_name="构美｜参考图编排", category="image/百炼 API/构美",
            inputs=[io.Combo.Input("scenario", options=SCENARIOS),
                    io.Image.Input("a_front"), io.Image.Input("a_back"), io.Image.Input("a_face"),
                    io.String.Input("extra_labels", default="[]", multiline=True,
                        tooltip='与额外图片逐张对应，如 ["A 外套胸前 logo 原始特写","背景参考"]。'),
                    io.Image.Input("b_front", optional=True), io.Image.Input("b_back", optional=True), io.Image.Input("b_face", optional=True),
                    io.Autogrow.Input("extras", template=io.Autogrow.TemplatePrefix(io.Image.Input("image", optional=True),
                        prefix="image", min=0, max=7), optional=True),
                    io.Int.Input("expected_count", default=0, min=0, max=10, optional=True,
                                 tooltip="0 按实际参考图数量；5 强制三张AI图＋两张原始细节图。"),
                    io.Custom("GOUMEI_DETAILS").Input("selected_details", optional=True)],
            outputs=[io.Image.Output("images"), io.String.Output("manifest")])

    @classmethod
    def execute(cls, scenario, a_front, a_back, a_face, extra_labels="[]", b_front=None, b_back=None, b_face=None, extras=None, expected_count=0, selected_details=None):
        tensors, labels = [], []
        roles = [("A", (a_front, a_back, a_face))]
        if scenario == "双人口播":
            roles.append(("B", (b_front, b_back, b_face)))
        for role, trio in roles:
            for purpose, batch in zip(("正面穿搭（颈部至鞋底）", "背面穿搭（颈部至鞋底）", "人物身份及人脸特写"), trio):
                if batch is None or len(batch) != 1:
                    raise ValueError(f"出镜者 {role} 的{purpose}须提供且仅有一张；生成节点请设置 n=1。")
                tensors.append(batch[0])
                labels.append(f"出镜者 {role}：{purpose}")
        extra_tensors = []
        for name in sorted(extras or {}, key=lambda name: int(name.removeprefix("image"))):
            if extras[name] is not None:
                extra_tensors.extend(extras[name])
        descriptions = json_value(extra_labels)
        if not isinstance(descriptions, list) or len(descriptions) != len(extra_tensors) or any(not isinstance(x, str) or not x.strip() for x in descriptions):
            raise ValueError("extra_labels 必须为与额外图片逐张对应的非空文字 JSON 数组。")
        tensors.extend(extra_tensors)
        labels.extend(descriptions)
        if selected_details is not None:
            tensors.extend(image[0] for image in selected_details["images"])
            labels.extend(selected_details["labels"])
        if len(tensors) > 10:
            raise ValueError(f"视频参考图最多 10 张，当前 {len(tensors)} 张；请调整素材，不会自动丢弃。")
        if expected_count and len(tensors) != expected_count:
            raise ValueError(f"本流程必须输入 {expected_count} 张参考图，实际 {len(tensors)} 张。")
        frames = []
        for tensor in tensors:
            image = Image.fromarray((tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
            # Batch tensors need equal dimensions; padding preserves garment and face proportions.
            image = ImageOps.contain(image, (1536, 1536), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1536, 1536), "white")
            canvas.paste(image, ((1536 - image.width) // 2, (1536 - image.height) // 2))
            frames.append(torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0))
        manifest = {"scenario": scenario, "images": [{"number": i, "purpose": label} for i, label in enumerate(labels, 1)]}
        return io.NodeOutput(torch.stack(frames), json.dumps(manifest, ensure_ascii=False, indent=2))


def dialogue_timeline(dialogue_json, duration, scenario, timing):
    entries = json_value(dialogue_json)
    if isinstance(entries, dict):
        entries = entries.get("segments")
    if not isinstance(entries, list) or not entries:
        raise ValueError('台词 JSON 必须包含非空 segments 数组或台词数组，例如 [{"role":"A","text":"你的台词"}]。')
    segments = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ("A", "B") or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise ValueError("每段台词须有 role（A/B）和非空 text；ASR 说话分组必须先映射到角色。")
        if scenario == "单人口播" and entry["role"] != "A":
            raise ValueError("单人口播仅允许 A 发言；有 B 时请选择双人口播或单人＋画外音。")
        item = {"role": entry["role"], "text": entry["text"].strip()}
        if timing == "保留确认时间":
            start, end = entry.get("start"), entry.get("end")
            if type(start) not in (int, float) or type(end) not in (int, float) or not math.isfinite(start + end) or not 0 <= start < end <= duration:
                raise ValueError("保留时间模式要求每段有合法 start/end 秒数，且在片段时长以内。")
            if segments and start < segments[-1]["end"]:
                raise ValueError("已确认台词时间不能重叠或倒序。")
            item.update(start=start, end=end)
        if segments and segments[-1]["role"] == item["role"] and (timing != "保留确认时间" or item["start"] == segments[-1]["end"]):
            segments[-1]["text"] += " " + item["text"]
            if "end" in item:
                segments[-1]["end"] = item["end"]
        else:
            segments.append(item)
    if timing != "保留确认时间":
        weights = [max(1, len(re.sub(r"\s", "", item["text"]))) for item in segments]
        position = 0
        for index, (item, weight) in enumerate(zip(segments, weights)):
            item["start"] = round(duration * position / sum(weights), 3)
            position += weight
            item["end"] = duration if index == len(segments) - 1 else round(duration * position / sum(weights), 3)
            if item["end"] <= item["start"]:
                raise ValueError("台词段数过多，无法分配有效时间。")
    return segments


class GoumeiDialogue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiDialogue", display_name="构美｜确认台词与音色路由", category="image/百炼 API/构美",
            inputs=[io.Combo.Input("scenario", options=SCENARIOS),
                    io.String.Input("dialogue_json", default="[]", multiline=True),
                    io.String.Input("video_info", force_input=True),
                    io.Combo.Input("timing", options=["按字数连续分配", "保留确认时间"]),
                    io.Boolean.Input("confirmed", default=False, tooltip="核对角色、台词与说话顺序后开启；未确认不会输出给视频生成。"),
                    io.Audio.Input("voice_a", optional=True), io.Audio.Input("voice_b", optional=True)],
            outputs=[io.String.Output("speech_prompt"), io.AnyType.Output("duration", tooltip="2–15 秒整数，可连接 Wan3 的旧版时长下拉输入。"), io.Audio.Output("audio_1"),
                     io.Audio.Output("audio_2"), io.String.Output("report")])

    @classmethod
    def execute(cls, scenario, dialogue_json, video_info, timing, confirmed, voice_a=None, voice_b=None):
        if not confirmed:
            raise ValueError("请先查看角色/台词分析，填写 dialogue_json 并开启 confirmed，之后再运行视频生成组。")
        source_duration = float(json_value(video_info)["duration"])
        if not math.isfinite(source_duration) or not 2 <= source_duration <= 15:
            raise ValueError("每段原片须为 2–15 秒；长视频先裁切片段，完成后再拼接。")
        duration = math.ceil(source_duration)
        segments = dialogue_timeline(dialogue_json, duration, scenario, timing)
        speaking = [role for role in ("A", "B") if any(item["role"] == role for item in segments)]
        voices = {"A": voice_a, "B": voice_b}
        for role in speaking:
            if voices[role] is None:
                raise ValueError(f"角色 {role} 有台词，必须连接该角色的参考音色。")
        audio_map = {role: index for index, role in enumerate(speaking, 1)}
        lines = [f"【角色与声音】\n场景：{scenario}。本段成片时长 {duration} 秒。"]
        for role in speaking:
            lines.append(f"角色 {role} 的全部台词只使用音频{audio_map[role]}的音色。")
        if scenario == "单人＋画外音":
            lines.append("仅 A 出镜，B 为画外音，始终不出镜，不生成人物 B。B 发言期间 A 自然倾听，不做说话口型。")
            if "A" not in speaking:
                lines.append("全片只有 B 的画外音，A 不发声、不做说话口型；只提供 B 的音频1。")
        elif scenario == "双人口播":
            lines.append("A、B 均出镜；各自发言时只同步自己的台词，倾听时不代说对方台词。")
        else:
            lines.append("仅 A 出镜并发言。")
        lines.append("不交换音色、不代说、不重叠、不重复台词。参考音频仅用于音色，不复述其内容。\n【已确认台词】时间从本段开头计算，逐字按以下顺序发言，不增删、不抢读下一句：")
        for item in segments:
            lines.append(f'{item["start"]:g}–{item["end"]:g}秒｜{item["role"]}，使用音频{audio_map[item["role"]]}的音色说：“{item["text"]}”')
        report = {"scenario": scenario, "duration": duration, "segments": segments, "audio_map": audio_map,
                  "actual_characters": sum(len(re.sub(r"\s", "", item["text"])) for item in segments),
                  "recommended_characters": [duration * 3, duration * 5]}
        routed = [voices[role] for role in speaking]
        return io.NodeOutput("\n".join(lines), duration, routed[0], routed[1] if len(routed) > 1 else None,
                             json.dumps(report, ensure_ascii=False, indent=2))


class GoumeiScenario(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiScenario", display_name="构美｜选择场景", category="image/百炼 API/构美",
            inputs=[io.Combo.Input("scenario", options=["单人口播", "单人＋画外音"], default="单人＋画外音")],
            outputs=[io.Combo.Output("scenario", options=SCENARIOS), io.String.Output("instructions")])

    @classmethod
    def execute(cls, scenario):
        if scenario not in ("单人口播", "单人＋画外音"):
            raise ValueError("此五图流程仅支持单人口播或单人＋画外音；双人请使用双人模板。")
        roles = "仅A出镜并发言，所有台词role为A。" if scenario == "单人口播" else "仅A出镜，B为画外音；根据口型和声音同步判断各句归属，允许只有B发言，不强行添加A的台词。"
        return io.NodeOutput(scenario, f"用户选择场景：{scenario}。{roles}\n自动分析素材、选择服饰、编排台词，无需人工确认。"
            "以五张服饰原图中整体图展示的主单品/成套穿搭为目标，细节图只补充对应同款；不要把不兼容款式混搭。"
            "只使用可见款式、颜色、结构等事实，不编造价格、优惠、成分比例、功效或原片商品特有事实。"
            "携带位置优先沿用原片可见的自然支撑关系，不凭平铺图推断左右。")


class GoumeiAutoDialogue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiAutoDialogue", display_name="构美｜自动台词与音色路由", category="image/百炼 API/构美",
            inputs=[io.Combo.Input("scenario", options=SCENARIOS), io.String.Input("dialogue_json", force_input=True),
                    io.String.Input("video_info", force_input=True), io.Audio.Input("voice_a", optional=True),
                    io.Audio.Input("voice_b", optional=True)], outputs=GoumeiDialogue.define_schema().outputs)

    @classmethod
    def execute(cls, scenario, dialogue_json, video_info, voice_a=None, voice_b=None):
        result = GoumeiDialogue.execute(scenario, dialogue_json, video_info, "按字数连续分配", True, voice_a, voice_b).result
        report = json_value(result[4])
        report["mode"] = "automatic"
        return io.NodeOutput(result[0].replace("【已确认台词】", "【自动编排台词】"), result[1], result[2], result[3],
                             json.dumps(report, ensure_ascii=False, indent=2))


class GoumeiVideoPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiVideoPrompt", display_name="构美｜组装视频提示词", category="image/百炼 API/构美",
            inputs=[io.String.Input("manifest", force_input=True), io.String.Input("visual_analysis", force_input=True),
                    io.String.Input("speech_prompt", force_input=True), io.String.Input("dialogue_report", force_input=True),
                    io.String.Input("face_hint", default="", optional=True, force_input=True),
                    io.Boolean.Input("captions_processed", default=True),
                    io.String.Input("scene_notes", default="背景沿用原片；未指定替换的商品与包装保持原样。", multiline=True)],
            outputs=[io.String.Output("prompt")])

    @classmethod
    def execute(cls, manifest, visual_analysis, speech_prompt, dialogue_report, captions_processed=True,
                scene_notes="背景沿用原片；未指定替换的商品与包装保持原样。", face_hint=""):
        mapping, dialogue = json_value(manifest), json_value(dialogue_report)
        if mapping["scenario"] != dialogue["scenario"]:
            raise ValueError("参考图编排与台词节点的场景类型必须一致。")
        text = [f"编辑视频1，生成{mapping['scenario']}。",
            "保留原片镜头顺序、运镜、人物行进路线和整体表演节奏。人物与目标服饰配饰的接触动作按目标穿戴或携带方式适配；冲突时调整手臂、手部和肩部姿态，不要求逐帧照搬原片手势。",
            "【角色与参考图对应】"]
        text.extend(f"图片{item['number']}：{item['purpose']}。" for item in mapping["images"])
        text.extend([
            "每位出镜者的正面、背面穿搭图定义同一套身体服饰，按原片朝向选用；人脸特写单独定义人物身份及头部外观。不混合原片人物五官，不继承不一致的旧衣物或配饰。转头、转身和遮挡后保持同一身份、肤色、五官和穿搭结构。",
            "【目标服饰及接触关系（以可见事实为准）】\n" + visual_analysis,
            "若参考图映射包含原始 LOGO／文字／花纹细节图，只补充对应单品的对应位置；生成三图中的标识或花纹若失真，以原始细节图为准。没有原始细节图时仅依据正面、背面和人脸参考，不补造标识、文字或花纹。不得新增第二件物品，不把局部图案复制到其他部位。",
            "目标单品跨镜头、转身和遮挡后保持一致，不残留旧部件、不叠加第二件。替换覆盖领口、袖口、手腕和遮挡边缘；移除超出目标轮廓的旧物并补全皮肤、衣物或背景。按目标款式与衣长还原，内搭仅在自然露出的位置显示，不强行改变开合。",
            "【穿戴与动作】左右以人物自身为准。携带方式或位置变化时移除旧位置物品及带子、提手，不在两处保留两件。手持物与握持点保持接触并受重力影响；肩背物保持支撑关系，转身和遮挡后延续。",
            "【场景与保留商品】\n" + scene_notes,
            "原片其余商品保持外观、包装和接触关系。按原片构图自然呈现，画外部位不额外展示，不生成拼图、参考展示板或文字说明。",
            speech_prompt])
        if captions_processed:
            text.append("视频1的字幕遮挡/模糊区域是输入预处理；成片补全自然画面，不保留遮挡块，不生成字幕。")
        if face_hint:
            text.append("视频1的人脸色块或模糊是输入预处理，按上述各角色人脸图还原清晰稳定的目标脸，不保留色块或模糊。\n" + face_hint)
        return io.NodeOutput("\n\n".join(text))


class GoumeiMuteVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="GoumeiMuteVideo", display_name="构美｜参考视频静音", category="image/百炼 API/构美",
            inputs=[io.Video.Input("video")], outputs=[io.Video.Output("video")])

    @classmethod
    def execute(cls, video):
        ffmpeg = os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("未找到 FFmpeg，请加入 PATH 或设置 FFMPEG_PATH。")
        work = Path(folder_paths.get_temp_directory()) / "goumei" / uuid.uuid4().hex
        work.mkdir(parents=True)
        source, destination = work / "source.mp4", work / "muted.mp4"
        video.save_to(str(source))
        result = subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", str(destination)],
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError("参考视频静音失败：" + result.stderr[-1500:])
        return io.NodeOutput(VideoFromFile(str(destination)))
