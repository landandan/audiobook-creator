import ast
import importlib
import os
from pathlib import Path
import unittest


class ChineseSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["BOOK_LANGUAGE"] = "zh"

        import utils.lang_config as lang_config
        import utils.run_shell_commands as run_shell_commands

        cls.lang_config = importlib.reload(lang_config)
        cls.run_shell_commands = importlib.reload(run_shell_commands)

    def test_chinese_chapter_paths_pass_command_validation(self):
        chapter_path = "temp_audio/第一章 陨落的天才.wav"

        self.assertTrue(
            self.run_shell_commands.validate_file_path_allowlist(chapter_path)
        )
        self.assertTrue(
            self.run_shell_commands.validate_command_arguments_allowlist(
                ["-i", chapter_path, "output.wav"]
            )
        )
        self.assertTrue(
            self.run_shell_commands.validate_command_safety(
                ["ffmpeg", "-i", chapter_path, "output.wav"]
            )
        )

    def test_command_validation_still_rejects_unsafe_paths(self):
        self.assertFalse(
            self.run_shell_commands.validate_command_arguments_allowlist(
                ["../第一章.wav"]
            )
        )
        self.assertFalse(
            self.run_shell_commands.validate_command_arguments_allowlist(
                ["第一章.wav;whoami"]
            )
        )

    def test_chinese_chapter_heading_detection(self):
        is_heading = self.lang_config.is_chinese_chapter_heading

        for heading in (
            "第一章 陨落的天才",
            "第一章\t重逢",
            "第一章陨落的天才",
            "第十二章：重逢",
            "第三百零五章",
            "番外篇：十年后",
        ):
            with self.subTest(heading=heading):
                self.assertTrue(is_heading(heading))

        for prose in (
            "第一章的内容很长",
            "第十二章他终于回来了",
            "第一章结束后，他离开了。",
        ):
            with self.subTest(prose=prose):
                self.assertFalse(is_heading(prose))

    def test_chinese_speaker_matching_prompt_is_used(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "utils"
            / "character_extraction_llm.py"
        )
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        matching_function = next(
            node
            for node in module.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "identify_speaker_for_dialogue_matching_only"
        )

        chinese_branch = next(
            node
            for node in ast.walk(matching_function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "is_chinese"
        )
        called_functions = {
            node.func.id
            for node in ast.walk(chinese_branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertIn("build_zh_speaker_matching_prompt", called_functions)


if __name__ == "__main__":
    unittest.main()
