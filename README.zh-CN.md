# Audiobook Creator

[English](README.md) | 简体中文

## 项目简介

Audiobook Creator 是一个开源的有声书制作工具，可将 EPUB、PDF、TXT 等格式的书籍转换为完整配音的有声书。项目结合大语言模型（LLM）与文本转语音（TTS）技术，能够识别小说人物、判断对话说话人，并为旁白和不同角色分配不同音色。

项目支持 [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) 和 [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)，提供异步并行生成、章节处理、多种音频格式以及 Gradio 图形界面。项目采用 GNU General Public License v3.0（GPL-3.0）许可证。

- [Orpheus TTS 多角色示例音频](https://audio.com/prakhar-sharma/audio/sample-orpheus-multi-voice-audiobook-orpheus)
- [Kokoro TTS 多角色示例音频](https://audio.com/prakhar-sharma/audio/generated-sample-multi-voice-audiobook-kokoro)

演示视频：

[![观看演示视频](https://img.youtube.com/vi/E5lUQoBjquo/maxresdefault.jpg)](https://www.youtube.com/watch?v=E5lUQoBjquo)

<details>
<summary>项目的四个主要组成部分</summary>

1. **文本清理与格式化（`book_to_txt.py`）**
   - 从 EPUB、PDF、TXT 等书籍文件中提取文本。
   - 规范特殊字符、换行和引号等格式。
   - 将结果保存为 `converted_book.txt`。

2. **人物识别与元数据生成（`identify_characters_and_output_book_to_jsonl.py`）**
   - 第一步：通过兼容 OpenAI API 的 LLM 识别人物及其年龄、性别等信息。
   - 第二步：根据上下文判断每段对话的说话人。
   - 生成以下文件：
     - `speaker_attributed_book.jsonl`：带说话人标注的文本。
     - `character_gender_map.json`：人物名称、年龄、性别和性别分数等元数据。

3. **情绪标签增强（`add_emotion_tags.py`）**
   - 添加 `<laugh>`、`<sigh>`、`<gasp>` 等情绪标签，提高朗读表现力。
   - 处理 `converted_book.txt`，输出 `tag_added_lines_chunks.txt`。
   - 该功能仅适用于 Orpheus TTS。

4. **有声书生成（`generate_audiobook.py`）**
   - 将清理后的文本或带说话人标注的文本转换为音频。
   - 支持 Kokoro 和 Orpheus 两种 TTS 引擎及其独立音色映射。
   - 提供两种朗读模式：
     - **单音色模式**：旁白使用一个音色，对话统一使用另一个音色。
     - **多音色模式**：根据人物属性为不同角色分配音色。
   - 输出到 `generated_audiobooks/audiobook.{输出格式}`。

</details>

## 主要功能

- 通过环境变量切换 Kokoro 或 Orpheus TTS。
- 异步并行调用 TTS，提高长篇书籍的生成速度。
- 提供易用的 Gradio Web 界面。
- 支持生成带封面、元数据和章节时间戳的 M4B 有声书。
- 支持 EPUB、PDF、TXT 等输入格式。
- 支持 AAC、M4A、MP3、WAV、OPUS、FLAC、PCM、M4B 输出格式。
- 支持 Docker 镜像和 Docker Compose。
- 使用 LLM 识别人物、人物属性和对话说话人。
- 支持单音色与多音色朗读以及旁白性别偏好。
- Orpheus TTS 支持情绪标签增强。
- 提供进度条和执行时间统计。
- 针对中文小说提供标点、引号、章节、人物称谓和中文音色支持。

## 示例文件

<details>
<summary>展开查看</summary>

`sample_book_and_audio` 目录包含：

- EPUB、PDF 和 TXT 格式的示例短篇小说。
- 清理后的 `converted_book.txt`。
- 带说话人标注的 `speaker_attributed_book.jsonl`。
- 人物元数据 `character_gender_map.json`。
- 使用 Kokoro 或 Orpheus 生成的单音色、多音色 MP3 和 M4B 示例。

</details>

## 运行要求

完整运行本项目需要：

- Docker Desktop，或者 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。
- 一个兼容 OpenAI API 的 LLM 服务，例如 LM Studio、vLLM、llama.cpp、SGLang 或云端模型服务。
- 一个独立运行的 TTS API 服务：Kokoro-FastAPI 或 Orpheus-TTS-FastAPI。
- FFmpeg：用于音频拼接和格式转换。
- Calibre（可选但推荐）：用于提高电子书格式兼容性；生成 M4B 时需要。
- 首次安装依赖、拉取镜像或下载模型时需要网络连接。

应用默认端口为 `7860`，Kokoro/Orpheus TTS 示例端口为 `8880`，LM Studio 常用端口为 `1234`。

## 快速开始

### 1. 准备 LLM 服务

启动一个兼容 OpenAI API 的模型服务。人物识别推荐使用 `Qwen/Qwen3-30B-A3B-Instruct-2507` 或同级别模型。

- 人物识别建议提供至少 20,000 token 的上下文窗口。
- 情绪标签建议提供至少 8,192 token 的上下文窗口。

### 2. 准备 TTS 服务

#### 方案一：Kokoro TTS（大多数用户推荐，中文必须使用）

通过 [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) 启动服务。

CUDA GPU：

```bash
docker run \
  --name kokoro_service \
  --restart always \
  --network host \
  --gpus all \
  ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.1 \
  uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
  --workers 2
```

CPU：

```bash
docker run \
  --name kokoro_service \
  --restart always \
  --network host \
  ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.1 \
  uvicorn api.src.main:app --host 0.0.0.0 --port 8880 --log-level debug \
  --workers 1
```

GPU 模式下，`--workers` 应与 `.env` 中的 `TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE` 保持一致。显存不足时请降低这两个值。

#### 方案二：Orpheus TTS（英文高质量与情绪标签）

按照 [Orpheus TTS FastAPI](https://github.com/prakharsr/Orpheus-TTS-FastAPI) 的说明部署服务。建议使用基于 vLLM 的 bf16、fp16 或 fp32 模型；量化模型可能产生重复、异常噪声、音频幻觉或无限循环。

Orpheus 目前仅建议用于英文书籍。中文小说请使用 Kokoro。

### 3. 配置环境变量

复制环境变量模板：

Linux/macOS：

```bash
cp .env_sample .env
```

Windows PowerShell：

```powershell
Copy-Item .env_sample .env
```

编辑 `.env`。变量值不要包裹单引号或双引号。

```dotenv
CHARACTER_IDENTIFICATION_LLM_BASE_URL=http://localhost:1234/v1
CHARACTER_IDENTIFICATION_LLM_API_KEY=lm-studio
CHARACTER_IDENTIFICATION_LLM_MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507

EMOTION_TAG_ADDITION_LLM_BASE_URL=http://localhost:1234/v1
EMOTION_TAG_ADDITION_LLM_API_KEY=lm-studio
EMOTION_TAG_ADDITION_LLM_MODEL_NAME=openai/gpt-oss-20b
EMOTION_TAG_ADDITION_LLM_MAX_PARALLEL_REQUESTS_BATCH_SIZE=1

TTS_BASE_URL=http://localhost:8880/v1
TTS_API_KEY=not-needed
TTS_MODEL=kokoro
TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE=2

NO_THINK_MODE=true
BOOK_LANGUAGE=zh
```

主要变量说明：

| 变量 | 说明 |
|---|---|
| `CHARACTER_IDENTIFICATION_LLM_*` | 人物识别和说话人归属使用的 LLM 地址、密钥和模型名 |
| `EMOTION_TAG_ADDITION_LLM_*` | 情绪标签使用的 LLM 配置，仅在相关流程中需要 |
| `TTS_BASE_URL` | TTS 服务的 OpenAI 兼容 API 地址 |
| `TTS_MODEL` | `kokoro` 或 `orpheus` |
| `TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE` | TTS 最大并发请求数 |
| `NO_THINK_MODE` | 是否要求支持的模型关闭思考模式 |
| `BOOK_LANGUAGE` | `zh` 表示中文，`en` 表示英文，默认 `en` |

### 4. 启动应用

#### 方式一：直接运行预构建 Docker 镜像

确保 `.env`、LLM 和 TTS 服务均已准备完成，然后执行：

```bash
docker run \
  --name audiobook_creator \
  --restart always \
  --network host \
  --env-file .env \
  ghcr.io/prakharsr/audiobook_creator:v2.0
```

访问 <http://localhost:7860>。

> 使用 Docker Desktop 时，需要在设置中开启 host networking。详情参见 [Docker host 网络说明](https://docs.docker.com/engine/network/drivers/host/)。

#### 方式二：Docker Compose

```bash
git clone https://github.com/prakharsr/audiobook-creator.git
cd audiobook-creator
docker compose --env-file .env up --build
```

访问 <http://localhost:7860>。

#### 方式三：使用 uv 直接运行

创建 Python 3.12 虚拟环境并安装依赖：

```bash
uv venv --python 3.12
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖并启动：

```bash
uv pip install -r requirements.txt --no-deps
uvicorn app:app --host 0.0.0.0 --port 7860
```

访问 <http://127.0.0.1:7860>。

如果需要更好的电子书解析或 M4B 输出，请安装 [Calibre](https://calibre-ebook.com/download) 并将其命令行工具加入 `PATH`。音频转换需要安装 [FFmpeg](https://www.ffmpeg.org/download.html)。

## 使用流程

1. 上传书籍并提取、清理文本。
2. 按需手动检查和编辑 `converted_book.txt`。
3. 如需多角色配音，运行人物识别和说话人归属。
4. 英文 Orpheus 用户可选用情绪标签增强。
5. 选择单音色或多音色、旁白性别和输出格式。
6. 生成结果将保存在 `generated_audiobooks` 目录。

## 中文小说支持

在 `.env` 中设置：

```dotenv
BOOK_LANGUAGE=zh
TTS_MODEL=kokoro
```

中文模式包含以下处理：

| 环节 | 中文模式行为 |
|---|---|
| 文本清洗 | 保留 `——`、`……`、`、`、`“”` 等中文标点 |
| 引号处理 | 将 `「…」`、`『…』` 归一为 `“…”`，并修复未配对引号 |
| 对话切分 | 识别中文弯引号、直角引号和英文直引号中的对话 |
| 人物识别 | 使用中文提示词处理全名、昵称、姓氏加称谓、职务称呼和代词线索 |
| 说话人归属 | 使用中文上下文、称谓和对话轮次判断说话人 |
| 章节识别 | 支持 `第X章`、`第X回`、`第X卷`、`第X部`、`第X节`、`第X集` |
| 特殊章节 | 支持 `楔子`、`序章`、`序言`、`尾声`、`终章`、`番外`、`番外篇` |
| 中文数字 | 支持“第十二章”“第三百零五章”等形式 |
| 中文文件名 | 章节音频文件保留中文字符，并安全传递给 FFmpeg |
| 中文音色 | 自动使用 `kokoro_zh` 音色映射并发送 `lang_code=z` |

推荐设置：

- 中文 TTS 使用 Kokoro；Orpheus 目前仅支持英文。
- 人物识别推荐 Qwen3 30B A3B Instruct 非思考模式。
- 男声旁白默认使用 `zm_yunjian`，女声旁白默认使用 `zf_xiaoxiao`。
- 可在 `static_files/voice_map.json` 的 `kokoro_zh` 节中自定义中文音色。
- 中文模式不使用 Orpheus 情绪标签。

## 上下文窗口与并发配置

### 人物识别

人物提取和逐行说话人归属需要较长上下文。非思考模型建议至少 20,000 token；思考模型通常需要更多。为了提高准确率，该步骤按顺序处理，可将 llama.cpp、vLLM 或 SGLang 的并行数设置为 `1` 以降低显存占用。

### 情绪标签

情绪标签请求建议至少提供 8,192 token 的上下文。各批次彼此独立，可以并行处理。使用 llama.cpp 时，上下文通常会在并发槽之间划分；例如并发数为 `8` 时，可将总上下文设置为 `8 × 8192 = 65536`。

## LLM 模型建议

- **人物识别**：Qwen3 30B A3B Instruct 非思考模式表现较好，适合人物发现和对话归属。
- **情绪标签**：带思考模式的 gpt-oss-20B 通常比 Qwen3 30B A3B Instruct 更适合添加情绪标签。

实际效果与小说类型、模型量化方式、提示词兼容性和推理速度有关，可以根据自己的硬件测试其他模型。

## 音频并行批量推理

`TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE` 控制同时向 TTS FastAPI 发出的最大请求数。

### Kokoro

- Docker 容器的 `--workers` 与环境变量建议设置为相同值。
- GPU 用户可从 `2` 开始尝试。
- 可根据显存近似从“显存 GB 数 ÷ 2”开始测试。
- 出现显存不足时，同时降低 worker 数和环境变量值。
- CPU 模式通常保持 `1` 即可。

### Orpheus

- 基于 vLLM 的服务支持异步并行处理。
- 一般可从 `4` 到 `8` 开始测试。
- 显存充足时可以尝试 `8` 到 `16`。
- 具体参数请参考 [Orpheus TTS FastAPI 文档](https://github.com/prakharsr/Orpheus-TTS-FastAPI)。

## 路线图

- ⏳ 支持在 Kokoro 和 Orpheus 已支持的更多语言之间切换。
- ✅ 使用两阶段 LLM 流程替代 Gliner NLP 人物识别流程。
- ✅ 支持 Orpheus 情绪标签增强。
- ✅ 支持 Kokoro 与 Orpheus TTS。
- ✅ 支持 TTS 批量并行推理。
- ✅ 支持男声或女声旁白。
- ✅ 支持 Docker 运行。
- ✅ 提供 Gradio UI。
- ✅ 支持 AAC、M4A、MP3、WAV、OPUS、FLAC、PCM、M4B。
- ✅ 使用 Calibre 提高文本、元数据和封面提取兼容性。
- ✅ 为 M4B 添加封面、章节和时间戳。
- ✅ 支持章节结尾停顿。
- ✅ 单音色模式下为旁白和对话使用不同音色。
- ✅ 仅将引号内的对话切换为角色音色。

## 支持

如有问题或建议，请在 [GitHub Issues](https://github.com/prakharsr/audiobook-creator/issues) 中提交。

## 许可证

本项目采用 GNU General Public License v3.0（GPL-3.0）许可证，详见 [LICENSE](LICENSE)。

## 参与贡献

欢迎通过 Issue 或 Pull Request 修复问题、改进文档或添加功能。

## 捐赠

如果本项目对你有所帮助，可以通过 [PayPal](https://paypal.me/prakharsr) 支持原作者。

---

祝你制作有声书顺利！如果项目对你有帮助，欢迎在 GitHub 上点亮 ⭐。
