"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import asyncio
import re
import random
import json
import os
import traceback
from typing import List, Dict, Set, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.exceptions import ModelHTTPError
from utils.llm_utils import check_if_have_to_include_no_think_token
from utils.file_utils import write_jsons_to_jsonl_file, empty_file, write_json_to_file
from utils.lang_config import get_dialogue_pattern, is_chinese
import tiktoken
from dotenv import load_dotenv

load_dotenv()

MAX_CONTEXT_WINDOW = int(os.environ.get("MAX_CONTEXT_WINDOW", "10240"))  # Total API context window
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "2000"))  # Tokens per batch (before overhead)
MAX_CONTEXT_TOKENS_PER_DIRECTION = int(os.environ.get("MAX_CONTEXT_TOKENS_PER_DIRECTION", "1000"))  # Tokens per direction
MAX_CONTEXT_LINES = 10  # Fallback line limit
TEMPERATURE = 0.2

PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.6"))  # OpenAI-style
FREQUENCY_PENALTY = float(os.environ.get("FREQUENCY_PENALTY", "0.3"))  # OpenAI-style
REPEAT_PENALTY = float(os.environ.get("REPEAT_PENALTY", "1.1"))  # llama.cpp native (1.0 = off)

_TOKENIZER = None

def get_tokenizer():
    """
    Get or initialize the global tokenizer.
    Uses cl100k_base encoding (GPT-4/GPT-3.5) as a good approximation for most LLMs.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo, etc.)
            # Provides good approximation for most modern LLMs
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
            print("✅ Initialized tiktoken (cl100k_base) for token counting")
            print(f"📊 Token Configuration:")
            print(f"   API Context Window: {MAX_CONTEXT_WINDOW} tokens")
            print(f"🔄 Repetition Prevention:")
            print(f"   Presence: {PRESENCE_PENALTY} | Frequency: {FREQUENCY_PENALTY} | Repeat: {REPEAT_PENALTY}")
        except Exception as e:
            print(f"⚠️  Failed to initialize tokenizer: {e}")
    return _TOKENIZER

def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken. Returns 0 if tokenizer unavailable.
    Note: Uses cl100k_base encoding as approximation - actual count may vary by model.
    """
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def is_context_overflow_error(error: Exception) -> bool:
    """
    Check if an error is due to context window overflow (typically from LLM repetition loops).
    
    Returns True if the error indicates context size exceeded.
    """
    if isinstance(error, ModelHTTPError):
        # Check for 400 status code and context size error messages
        if error.status_code == 400:
            error_msg = str(error).lower()
            return any(phrase in error_msg for phrase in [
                'exceed_context_size',
                'context size',
                'context length',
                'too many tokens',
                'maximum context'
            ])
    return False


# Pydantic models for structured outputs

class CharacterInfo(BaseModel):
    """Information about a single character with explicit operation type."""
    reason: str = Field(description="Brief explanation: 1) What mentions/clues found, 2) Operation type justification, 3) Key distinguishing details")
    operation: Literal["insert", "update", "merge"] = Field(description="Operation type: MUST be exactly 'insert' (brand new character), 'update' (same name, add info), or 'merge' (consolidate old_name into new canonical name)")
    name: str = Field(description="Canonical character name (lowercase) - use most complete form available")
    existing_name: Optional[str] = Field(default=None, description="REQUIRED for 'merge' operations ONLY: the existing character name to consolidate (e.g., 'mr. dursley' when name is 'vernon dursley'). MUST be null for 'insert' and 'update'.")
    age: Literal["child", "adult", "elderly"] = Field(description="Age category: MUST be exactly 'child', 'adult', or 'elderly'")
    gender: Literal["male", "female", "unknown"] = Field(description="Gender: MUST be exactly 'male', 'female', or 'unknown'")
    description: str = Field(description="Brief 2-3 sentence description with distinguishing features")


class CharacterExtractionResult(BaseModel):
    """Result from character information extraction."""
    characters: List[CharacterInfo] = Field(
        default_factory=list,
        description="List of characters found or updated in the text"
    )


class SpeakerAttributionResult(BaseModel):
    """Result from dialogue speaker attribution."""
    reasoning: str = Field(description="Structured explanation of how the speaker was identified. Include: 1) Dialogue tag scan results, 2) Character matching process, 3) Final decision rationale")
    speaker: str = Field(description="Canonical character name who is speaking (lowercase)")
    is_new_character: bool = Field(description="Whether this is a newly identified character")
    character_info: Optional[CharacterInfo] = Field(
        default=None,
        description="Character information if this is a new character"
    )


def has_dialogue(line: str) -> bool:
    """Check if a line contains dialogue (text in quotes, language-aware)."""
    return bool(get_dialogue_pattern().search(line))


def preprocess_text_into_lines(text: str) -> List[Dict]:
    """
    Preprocess text by splitting lines with mixed dialogue/narrator content into separate lines.
    This simplifies downstream processing by making everything a simple sequence.
    
    Args:
        text: The full book text
        
    Returns:
        List of dicts with structure:
        {
            "type": "narrator" | "dialogue",
            "content": str,
            "original_line_idx": int,  # Which original line this came from
            "original_line": str        # The full original line (for context)
        }
    
    Example:
        Input line: 'He said, "Hello," and walked away.'
        Output: [
            {"type": "narrator", "content": "He said,", "original_line_idx": 5, "original_line": 'He said...'},
            {"type": "dialogue", "content": '"Hello,"', "original_line_idx": 5, "original_line": 'He said...'},
            {"type": "narrator", "content": "and walked away.", "original_line_idx": 5, "original_line": 'He said...'}
        ]
    """
    original_lines = text.split("\n")
    processed_items = []
    
    for line_idx, line in enumerate(original_lines):
        # Empty lines remain as narrator lines
        if not line.strip():
            processed_items.append({
                "type": "narrator",
                "content": line,
                "original_line_idx": line_idx,
                "original_line": line
            })
            continue
        
        # Check if line has dialogue
        if has_dialogue(line):
            # Split into parts
            last_end = 0

            for match in get_dialogue_pattern().finditer(line):
                start, end = match.span()
                
                # Add narrator text before this dialogue (if any)
                if start > last_end:
                    narrator_text = line[last_end:start].strip()
                    if narrator_text:
                        processed_items.append({
                            "type": "narrator",
                            "content": narrator_text,
                            "original_line_idx": line_idx,
                            "original_line": line
                        })
                
                # Add the dialogue itself
                dialogue_text = match.group(0)
                processed_items.append({
                    "type": "dialogue",
                    "content": dialogue_text,
                    "original_line_idx": line_idx,
                    "original_line": line
                })
                
                last_end = end
            
            # Add any remaining narrator text after the last dialogue
            if last_end < len(line):
                narrator_text = line[last_end:].strip()
                if narrator_text:
                    processed_items.append({
                        "type": "narrator",
                        "content": narrator_text,
                        "original_line_idx": line_idx,
                        "original_line": line
                    })
        else:
            # Pure narrator line
            processed_items.append({
                "type": "narrator",
                "content": line,
                "original_line_idx": line_idx,
                "original_line": line
            })
    
    return processed_items


def calculate_gender_score(age: str, gender: str) -> int:
    """
    Calculate gender score based on age and gender.
    Returns a score from 1-10 where 1 is completely masculine and 10 is completely feminine.
    """
    gender_score = 5  # Default to neutral/unknown
    
    if gender == "male":
        if age == "child":
            gender_score = 4  # Slightly masculine for male children
        elif age == "adult":
            gender_score = random.choice([1, 2, 3])  # Mostly to completely masculine for male adults
        elif age == "elderly":
            gender_score = random.choice([1, 2])  # Mostly to completely masculine for elderly males
    elif gender == "female":
        if age == "child":
            gender_score = 10  # Completely feminine for female children
        elif age == "adult":
            gender_score = random.choice([7, 8, 9])  # Mostly to completely feminine for female adults
        elif age == "elderly":
            gender_score = random.choice([6, 7])  # Slightly to moderately feminine for elderly females
    
    return gender_score


def get_context_for_item(
    processed_items: List[Dict],
    current_idx: int,
    line_map: List[Dict],
    max_tokens: int = MAX_CONTEXT_TOKENS_PER_DIRECTION,
    items_limit: int = MAX_CONTEXT_LINES * 3  # More items since lines are split
) -> Tuple[str, str]:
    """
    Get context window around a dialogue item with speaker annotations.
    Works with preprocessed items from preprocess_text_into_lines().
    
    Args:
        processed_items: List of preprocessed items
        current_idx: Index of current dialogue item
        line_map: Already processed items with speaker attributions
        max_tokens: Maximum tokens for context in each direction
        items_limit: Maximum number of items to include
    
    Returns:
        Tuple of (context_before_text, context_after_text) as formatted strings
    """
    context_before_parts = []
    context_after_parts = []
    
    # Get context before (working backwards)
    token_count = 0
    for i in range(current_idx - 1, max(0, current_idx - items_limit) - 1, -1):
        if i < 0:
            break
        
        item = processed_items[i]
        content = item["content"]
        
        # Format with speaker info if already processed
        if i < len(line_map):
            speaker = line_map[i].get("speaker", "narrator")
            if speaker != "narrator" and item["type"] == "dialogue":
                formatted = f"[{speaker}]: {content}"
            else:
                formatted = content
        else:
            formatted = content
        
        # Count tokens
        line_tokens = count_tokens(formatted)
        if token_count + line_tokens > max_tokens:
            break
        
        context_before_parts.insert(0, formatted)
        token_count += line_tokens
    
    # Get context after (working forwards)
    token_count = 0
    for i in range(current_idx + 1, min(len(processed_items), current_idx + items_limit + 1)):
        if i >= len(processed_items):
            break
        
        item = processed_items[i]
        content = item["content"]
        
        # Count tokens
        line_tokens = count_tokens(content)
        if token_count + line_tokens > max_tokens:
            break
        
        context_after_parts.append(content)
        token_count += line_tokens
    
    context_before_text = "\n".join(context_before_parts) if context_before_parts else "[No previous context]"
    context_after_text = "\n".join(context_after_parts) if context_after_parts else "[No following context]"
    
    return context_before_text, context_after_text


def format_character_map_for_prompt(character_map: Dict[str, Dict]) -> str:
    """Format character map as a readable string for LLM prompts."""
    character_list = []
    for char_name, char_info in character_map.items():
        character_list.append(
            f"- {char_name}: {char_info.get('description', 'No description yet')} "
            f"(Age: {char_info.get('age', 'unknown')}, Gender: {char_info.get('gender', 'unknown')})"
        )
    return "\n".join(character_list) if character_list else "No characters identified yet."


def build_zh_character_extraction_prompt(no_think_token: str, character_context: str) -> str:
    """Chinese system prompt for Pass 1 character extraction (BOOK_LANGUAGE=zh)."""
    return f"""{no_think_token}
你是一位中文小说人物分析专家。你的任务是从小说文本中识别新登场的人物，或为已知人物补充新信息。

重要：只提取"人物"和"会说话的类人角色"（如有名字且会说话的器灵、精怪、灵魂体）。
不要提取地点、物品、门派/势力、功法、事件或不会说话的动物——这些都不需要配音。

当前已知人物：
{character_context}

关键规则：

1. **先排查是否为已知人物**——中文小说里同一个人常有多种称呼：
   - 全名 vs 单称/昵称（"萧炎" vs "炎儿"）
   - 姓氏+称谓（"王老师" 可能就是 "王岚"）
   - 职务/身份称呼（"萧族长" = "萧战"）
   - 绰号（"药老" = "药尘"）
   - 描述性指代（"黑衣老者"可能是已知的某位老人角色）

   **创建新人物前必须：**
   - 逐一比对已知人物名单，判断当前称呼是否指向同一人
   - 注意：同姓通常是不同人（"萧战"与"萧炎"是两个人），需结合上下文关系判断
   - 若有 70% 以上把握是同一人，用 update 而非新建；拿不准时宁 update 勿 insert

2. **操作类型**（operation 只能严格是这三个值之一，其他值会导致校验失败并重试）：
   - "insert"：全新人物首次登场，existing_name=null
   - "update"：同一人、同一规范名，补充新信息，existing_name=null
   - "merge"：不同名字实为同一人，name=更完整的规范名，existing_name=要合并掉的旧名
     注意：existing_name 必须是已知名单中真实存在的名字，且与 name 不同！同名请用 update，新人物请用 insert。

   规范名优先级：全名 > 姓氏+称谓 > 职务称呼 > 绰号/昵称

3. **字段校验**：
   - age 只能是 "child"、"adult"、"elderly" 之一
   - gender 只能是 "male"、"female"、"unknown" 之一
   - 性别判断线索：代词（他/她）、称谓（公子/姑娘/小姐/夫人/少年/老者/前辈）、名字与描写
   - 拿不准时使用 age="adult", gender="unknown"

4. **description 质量**：用中文写 2-3 句，包含身份、与其他人物的关系、外貌/性格/功法等特征，供后续识别同一人的不同称呼。

5. **reason 字段格式**（先推理再填其他字段）：
   1. FOUND：在文中发现了什么名字/称呼？
   2. CHECK：是否匹配某个已知人物？是哪一个？
   3. OPERATION：insert（全新）/ update（同名补充）/ merge（旧名→新规范名）

6. **什么算人物**：
   - 提取：有名字的人、有名字且会说话的类人存在
   - 不提取：地点（如"乌坦城"）、门派势力（如"云岚宗"）、功法物品、不会说话的兽类、无名字的泛指（"一个路人"、"众人"）
   - 判断标准："这个角色需要配音演员吗？"不需要就不提取。
   - 拿不准时倾向于提取，宁可多提也不要漏提。"""


def build_zh_speaker_matching_prompt(no_think_token: str, character_context: str, exact_names_list: str) -> str:
    """Chinese system prompt for Pass 2 speaker attribution (BOOK_LANGUAGE=zh)."""
    return f"""{no_think_token}
你是中文小说对话归属专家。你的唯一任务是判断给定对话是哪位已知人物说的。

所有人物已在上一轮提取完毕，你只做匹配——绝不创建新人物。

已知人物（完整名单，所有人都在此）：
{character_context}

关键规则：

1. **必须逐字复制名字**：
   - 返回值必须与下方有效名单中的名字一字不差
   - 不得增加、删减或改动任何字，不得使用名单外的称呼

2. **有效说话人名单（只能从中复制）**：
{exact_names_list}

3. **匹配流程**：
   - 首先查看 context_before 中 [CURRENT LINE WITH DIALOGUE TAGS] 一行的说话提示，如"萧炎说道"、"药老抚须笑道"、"她轻声道"
   - 将提示中的名字/称谓/代词与上方人物描述比对
   - 利用 [speaker]: 标记追踪对话轮次，判断当前接话的人
   - 代词线索："他/她" + 上下文最近互动的异性/同性人物 + 性别年龄信息
   - 对话内称呼线索："老师"、"父亲"、"炎儿"等可推断说话双方身份

4. **特殊情况**：
   - 若这段"对话"实为叙述文字（引号误标）→ 返回 "narrator"
   - 若确实无法判断 → 返回 "unknown"
   - 绝不编造名单之外的名字

5. **校验要求**：你的返回值会对照有效名单校验，不在名单中会报错重试。"""


async def extract_character_info_from_batch(
    batch_lines: List[str],
    character_map: Dict[str, Dict],
    async_openai_client,
    model_name: str
) -> Dict[str, Dict]:
    """
    Extract character information from a batch of non-dialogue lines using Pydantic AI.
    Returns updates to character_map (new characters or updated descriptions).
    """
    try:
        batch_text = "\n".join(batch_lines)
        no_think_token = check_if_have_to_include_no_think_token()
        
        # Format current character map for LLM
        character_context = format_character_map_for_prompt(character_map)
        
        # Create system prompt
        system_prompt = f"""{no_think_token}
You are an expert at analyzing narrative text and extracting character information. Your task is to identify any NEW characters introduced in the text or find NEW information about EXISTING characters.

IMPORTANT: ONLY extract PEOPLE and SPEAKING HUMANOID BEINGS.
DO NOT extract locations, objects, institutions, events, or animals - these are not characters who speak dialogue.

Current known characters:
{character_context}

CRITICAL INSTRUCTIONS FOR CHARACTER IDENTIFICATION:

1. **Check for Existing Characters FIRST** - Before adding a new character, carefully check if they might already exist under a different name:
   - Full names vs. surnames only (e.g., "Vernon Dursley" = "Mr. Dursley" = "Dursley")
   - Titles with names (e.g., "Professor McGonagall" = "McGonagall")
   - First names vs. full names (e.g., "Harry" = "Harry Potter")
   - Familial titles (e.g., "Uncle Vernon" = "Vernon Dursley")
   - Nicknames and informal references (e.g., "Hermione" = "Hermione Granger")
   - Descriptive references to known characters (e.g., "the old man" might be an already-known elderly character)
   
   **BEFORE creating ANY new character, you MUST:**
   - Check EVERY existing character name to see if current reference could match
   - Look for substring matches (e.g., "Dursley" matches "Vernon Dursley")
   - Look for name variations (e.g., "centaur" might match existing character described as centaur)
   - Check if title + surname matches full name (e.g., "Mr. Dursley" = "Vernon Dursley")
   - If >70% confident it's the same person, UPDATE the existing character instead
   - When in doubt, UPDATE rather than CREATE to avoid duplicates
   
2. **Operation Types** - You MUST specify EXACTLY one of these three operations for each character:
   CRITICAL: operation field accepts ONLY these exact strings: "insert", "update", "merge"
   Any other value (like "create", "add", "new") will cause validation failure and retry!
   
   **INSERT** - Brand new character not in known list:
   - Use when: Character appears for first time
   - Set: operation="insert", existing_name=null
   - Example: Found "Hagrid" and no similar character exists -> INSERT "hagrid"
   
   **UPDATE** - Same character, same name, new information:
   - Use when: Adding details to an existing character (same canonical name)
   - Set: operation="update", name=[existing name], existing_name=None
   - Example: "vernon dursley" exists, found more description -> UPDATE "vernon dursley"
   
   **MERGE** - Different names, same person (consolidate to most complete name):
   - Use when: Found a more complete/better name for an existing character
   - Set: operation="merge", name=[NEW canonical name], existing_name=[OLD name to merge from]
   - Example: Have "mr. dursley", found "Vernon Dursley" -> MERGE existing_name="mr. dursley" into name="vernon dursley"
   - Example: Have "dumbledore", found "Albus Dumbledore" -> MERGE existing_name="dumbledore" into name="albus dumbledore"
   - CRITICAL: existing_name MUST be a VALID character name from the known list!
   - CRITICAL: name and existing_name MUST be DIFFERENT! If same name, use UPDATE instead!
   - WRONG: MERGE existing_name=None or existing_name="none" or existing_name="null" (use INSERT!)
   - WRONG: MERGE existing_name="aunt petunia" into name="aunt petunia" (use UPDATE!)
   - CORRECT: MERGE existing_name="mr. dursley" into name="vernon dursley" (different names, both valid)
   - CORRECT: UPDATE name="aunt petunia" (same name = UPDATE, not MERGE)
   - CORRECT: INSERT name="new character" (brand new = INSERT, not MERGE)
   
   **Name Priority for MERGE**:
   - Full name > Partial name ("vernon dursley" > "mr. dursley")
   - With first name > Title only ("harry potter" > "mr. potter")
   - Keep names lowercase for consistency

3. **Character Extraction Process**:
   - Actively look for character introductions, names, physical descriptions, age indicators, gender clues, and relationships
   - Extract from BOTH narrative text AND dialogue (e.g., '"I'm Hagrid" said the giant' -> extract "hagrid")
   - Self-introductions in dialogue are important (e.g., '"My name is..."', '"Call me..."', '"I am..."')
   - For NEW characters: extract name (most complete form available), brief description (2-3 sentences max)
   - For EXISTING characters: only update if you find significant NEW information or a more complete name form
   - **Be proactive**: If someone has a name and any context about them, extract them as a character
   - Names with descriptions are ALWAYS characters (e.g., "Mr. Dursley was the director of a firm" -> extract "mr. dursley")
   - Dialogue tags with names are ALWAYS characters (e.g., '"Hello," said Mr. Ollivander' -> extract "mr. ollivander")
   
   **CRITICAL FIELD VALIDATION**:
   - age: MUST be EXACTLY "child", "adult", or "elderly" (no other values accepted!)
   - gender: MUST be EXACTLY "male", "female", or "unknown" (no other values accepted!)
   - If unsure, use: age="adult", gender="unknown"
   
4. **Description Quality**:
   - Keep descriptions brief but distinctive
   - Include relationships to other characters (e.g., "Harry's uncle", "Hermione's friend")
   - Include physical traits, personality, or role if mentioned
   - These descriptions help identify the same character referenced differently later

5. **Output Requirements & Reasoning Format**:
   - Return ALL characters that appear or are referenced in this text batch with their names
   - ONLY return empty list if the text is truly pure description/setting with no named individuals
   - **When in doubt, extract the character** - it's better to extract than to miss
   
   **CRITICAL - Reasoning Structure (think through this FIRST before filling other fields):**
   For each character, your 'reason' field MUST follow this format:
   ```
   1. FOUND: [What name/reference did you find in text?]
   2. CHECK: [Does this match any existing character? Which one?]
   3. OPERATION: [INSERT (new) / UPDATE (same name, new info) / MERGE (old_name -> new_name)]
   ```
   
   Examples:
   - INSERT: "1. FOUND: 'Hagrid' with description. 2. CHECK: No existing Hagrid found. 3. OPERATION: INSERT 'hagrid'"
   - UPDATE: "1. FOUND: More details about Vernon Dursley. 2. CHECK: Matches existing 'vernon dursley'. 3. OPERATION: UPDATE 'vernon dursley'"
   - MERGE: "1. FOUND: 'Vernon Dursley' is full name. 2. CHECK: Matches existing 'mr. dursley'. 3. OPERATION: MERGE 'mr. dursley' -> 'vernon dursley'"

6. **What Qualifies as a Character**:
   **EXTRACT these as characters:**
   - Human beings with proper names (e.g., "Harry", "Mr. Dursley", "Professor McGonagall")
   - Humanoid beings with names who can speak (e.g., ghosts, centaurs, named goblins)
   - Named individuals with descriptions (e.g., "Mr. Dursley was the director...")
   - Family members mentioned by name or title (e.g., "Mrs. Dursley", "their son Dudley")
   
   **DO NOT EXTRACT these (they are NOT speaking characters):**
   - LOCATIONS: places, buildings, rooms, streets, stations (e.g., "Diagon Alley", "King's Cross Station", "Great Hall", "Leaky Cauldron", "Privet Drive", "library", "kitchens", "corridors", "Platform 9 3/4")
   - OBJECTS: items, tools, clothing, furniture, vehicles (e.g., "Hogwarts Express", "Mirror of Erised", "Philosopher's Stone", "Invisibility Cloak", "Daily Prophet", "put-outer")
   - INSTITUTIONS: organizations, banks, governments, schools (e.g., "Gringotts", "Ministry of Magic")
   - EVENTS/ACTIVITIES: games, fights, meals, ceremonies (e.g., "chess game", "snowball fight", "Christmas dinner")
   - ANIMALS/PETS: non-speaking creatures (e.g., "Hedwig", "Fang", "Norbert", "Mrs. Norris", "Scabbers")
   
   **Test: "Would this need a voice actor to speak dialogue?" If NO, do not extract it.**

OPERATION EXAMPLES (follow these patterns):

**INSERT Examples**:
- Text: "Mr. Dursley was the director..." + no Dursley exists -> INSERT operation="insert", name="mr. dursley"
- Text: "Professor Dumbledore arrived" + no Dumbledore exists -> INSERT operation="insert", name="professor dumbledore"
- Text: "Hagrid stepped through the door" + no Hagrid exists -> INSERT operation="insert", name="hagrid"

**UPDATE Examples**:
- Have "vernon dursley", text adds more description -> UPDATE operation="update", name="vernon dursley"
- Have "harry potter", text mentions his age/appearance -> UPDATE operation="update", name="harry potter"

**MERGE Examples** (most important for accuracy):
- Have "mr. dursley", found "Vernon Dursley" -> MERGE operation="merge", name="vernon dursley", existing_name="mr. dursley"
- Have "dumbledore", found "Albus Dumbledore" -> MERGE operation="merge", name="albus dumbledore", existing_name="dumbledore"
- Have "professor mcgonagall", found "Minerva McGonagall" -> MERGE operation="merge", name="minerva mcgonagall", existing_name="professor mcgonagall"
- Have "uncle vernon", found "Vernon Dursley" -> MERGE operation="merge", name="vernon dursley", existing_name="uncle vernon"

**DON'T Extract**:
- Generic references without names: "the boy", "a man", "someone" (no proper name given)
- Locations, objects, institutions, events, or animals (see section 6 above for detailed examples)

ANTI-DUPLICATE EXAMPLES (use MERGE for these):
- "Malfoy" when "Draco Malfoy" exists -> MERGE operation="merge", name="draco malfoy", existing_name="malfoy" (if "malfoy" was extracted earlier)
- "Dumbledore" when "Albus Dumbledore" exists -> MERGE operation="merge", name="albus dumbledore", existing_name="dumbledore"
- "Neville" when "Neville Longbottom" exists -> MERGE operation="merge", name="neville longbottom", existing_name="neville"
- Don't create generic references: "the twins", "a centaur", "the boy" (no proper names)"""
        
        # Create user prompt
        user_prompt = f"""Analyze this text excerpt and extract character information:

<text_excerpt>
{batch_text}
</text_excerpt>"""

        # Chinese books: switch to Chinese prompts - Chinese address forms,
        # nicknames and pronoun gender cues differ fundamentally from English.
        if is_chinese():
            system_prompt = build_zh_character_extraction_prompt(no_think_token, character_context)
            user_prompt = f"""请分析以下中文小说片段，提取其中的人物信息：

<text_excerpt>
{batch_text}
</text_excerpt>"""
        
        # Count tokens for monitoring
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_prompt)
        total_input_tokens = system_tokens + user_tokens

        # Manual retry logic with escalating penalties for context overflow (LLM repetition loops)
        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            try:
                # Calculate escalating penalties for this attempt
                current_presence = PRESENCE_PENALTY + (attempt * 0.2)  # +0.2 per retry
                current_frequency = FREQUENCY_PENALTY + (attempt * 0.2)  # +0.2 per retry
                current_repeat = REPEAT_PENALTY + (attempt * 0.05)  # +0.05 per retry
                
                if attempt > 0:
                    print(f"🔄 Character Extraction - Retry attempt {attempt + 1}/{max_retries}")
                    print(f"   Increasing penalties: presence={current_presence:.2f}, frequency={current_frequency:.2f}, repeat={current_repeat:.2f}")
                
                # Create Pydantic AI agent with structured output
                provider = OpenAIProvider(openai_client=async_openai_client)
                model = OpenAIChatModel(model_name, provider=provider)
                agent = Agent(
                    model=model,
                    output_type=CharacterExtractionResult,
                    retries=3,  # Auto-retry on validation failures (e.g., invalid enum values)
                    # Pydantic AI will automatically retry if LLM outputs:
                    # - Invalid operation (not "insert"/"update"/"merge")
                    # - Invalid age (not "child"/"adult"/"elderly")
                    # - Invalid gender (not "male"/"female"/"unknown")
                    # It sends the validation error back to LLM to fix!
                    model_settings={
                        "temperature": TEMPERATURE,
                        "presence_penalty": current_presence,
                        "frequency_penalty": current_frequency,
                        "repeat_penalty": current_repeat
                    },
                    system_prompt=system_prompt
                )
                
                # Run agent to get structured output
                result = await agent.run(user_prompt)
                
                token_usage = result.usage()
                # print("--------------------------------")
                # output_dict = json.loads(result.output.model_dump_json())
                # print(json.dumps({
                #     "batch_text": batch_text,
                #     **output_dict
                # }, indent=4))
                print(f"📊 Character Extraction - Usage: {token_usage.input_tokens} input, {token_usage.output_tokens} output, {token_usage.total_tokens} total")
                
                # Success! Break out of retry loop
                break
                
            except Exception as e:
                if is_context_overflow_error(e):
                    if attempt < max_retries - 1:
                        print(f"⚠️  Context overflow detected (likely LLM repetition loop). Retrying with higher penalties...")
                        continue
                    else:
                        print(f"❌ Context overflow persisted after {max_retries} retries. Giving up.")
                        raise
                else:
                    # Different error, don't retry
                    raise
        
        if result is None:
            raise Exception("Failed to get result after retries")
    
        # Process structured output with explicit operation handling
        extraction_result: CharacterExtractionResult = result.output

        updates = {}
        merges = {}  # Track merge operations: old_name → new_name

        for char in extraction_result.characters:
            char_name = char.name.strip().lower()
            if not char_name:
                continue
            
            # No validation needed - Literal types guarantee valid values!
            operation = char.operation  # Guaranteed to be "insert", "update", or "merge"
            age = char.age  # Guaranteed to be "child", "adult", or "elderly"
            gender = char.gender  # Guaranteed to be "male", "female", or "unknown"
            
            if operation == "insert":
                # Brand new character
                print(f"✨ INSERT: New character '{char_name}'")
                updates[char_name] = {
                    "name": char_name,
                    "age": age,
                    "gender": gender,
                    "gender_score": calculate_gender_score(age, gender),
                    "description": char.description
                }
                
            elif operation == "update":
                # Same character, new information
                print(f"🔄 UPDATE: Updating '{char_name}' with new information")
                if char_name in character_map:
                    existing = character_map[char_name]
                    # Keep existing demographics unless new ones are more specific
                    if existing.get("age") != "unknown":
                        age = existing.get("age", age)
                    if existing.get("gender") != "unknown":
                        gender = existing.get("gender", gender)
                    # Use longer/more detailed description
                    existing_desc = existing.get("description", "")
                    description = char.description if len(char.description) > len(existing_desc) else existing_desc
                else:
                    description = char.description
                    
                updates[char_name] = {
                    "name": char_name,
                    "age": age,
                    "gender": gender,
                    "gender_score": calculate_gender_score(age, gender),
                    "description": description
                }
                
            elif operation == "merge":
                # Different name, same person - consolidate
                old_name = char.existing_name.strip().lower() if char.existing_name else None
                
                # Validation 1: Skip if existing_name is None, empty, or literal "none"/"null"
                if not old_name or old_name in ["none", "null", "n/a", "unknown"]:
                    print(f"⚠️  MERGE for '{char_name}' has invalid existing_name='{char.existing_name}'. Skipping this character.")
                    print(f"    (MERGE requires a valid old character name. LLM should use INSERT instead.)")
                    continue  # Skip this character entirely
                
                # Validation 2: Check for self-merge (same name → same name)
                if old_name == char_name:
                    print(f"⚠️  MERGE with same name '{char_name}' → '{char_name}' detected. Converting to UPDATE.")
                    # Treat as UPDATE instead
                    if char_name in character_map:
                        existing = character_map[char_name]
                        # Keep existing demographics unless new ones are more specific
                        if existing.get("age") != "unknown":
                            age = existing.get("age", age)
                        if existing.get("gender") != "unknown":
                            gender = existing.get("gender", gender)
                        # Use longer/more detailed description
                        existing_desc = existing.get("description", "")
                        description = char.description if len(char.description) > len(existing_desc) else existing_desc
                    else:
                        description = char.description
                
                elif old_name and old_name in character_map:
                    print(f"🔀 MERGE: '{old_name}' → '{char_name}' (consolidating to canonical name)")
                    merges[old_name] = char_name
                    
                    # Merge information from old character
                    existing = character_map[old_name]
                    # Prefer new demographics if more specific
                    if existing.get("age") != "unknown" and age == "unknown":
                        age = existing.get("age")
                    if existing.get("gender") != "unknown" and gender == "unknown":
                        gender = existing.get("gender")
                    # Combine descriptions
                    existing_desc = existing.get("description", "")
                    new_desc = char.description
                    description = new_desc if len(new_desc) > len(existing_desc) else f"{existing_desc} {new_desc}".strip()
                else:
                    print(f"⚠️  MERGE requested for '{old_name}' → '{char_name}', but '{old_name}' not found. Skipping this character.")
                    continue  # Skip instead of treating as INSERT
                    
                updates[char_name] = {
                    "name": char_name,
                    "age": age,
                    "gender": gender,
                    "gender_score": calculate_gender_score(age, gender),
                    "description": description
                }
                
            else:
                print(f"⚠️  Unknown operation '{operation}' for '{char_name}'. Treating as INSERT.")
                updates[char_name] = {
                    "name": char_name,
                    "age": age,
                    "gender": gender,
                    "gender_score": calculate_gender_score(age, gender),
                    "description": char.description
                }
        
        # Remove merged characters (old names) from updates
        for old_name in merges.keys():
            if old_name in updates:
                del updates[old_name]
                print(f"🗑️  Removed old name '{old_name}' after merge")
        
        return updates, merges
    
    except Exception as e:
        print(f"Error extracting character info: {e}")
        print(f"\n{'='*80}")
        print(f"SYSTEM PROMPT:")
        print(f"{'='*80}")
        print(system_prompt)
        print(f"\n{'='*80}")
        print(f"USER PROMPT:")
        print(f"{'='*80}")
        print(user_prompt)
        print(f"{'='*80}\n")
        traceback.print_exc()
        return {}


async def identify_speaker_for_dialogue_matching_only(
    dialogue_line: str,
    context_before: str,
    context_after: str,
    character_map: Dict[str, Dict],
    async_openai_client,
    model_name: str
) -> str:
    """
    PASS 2 VERSION: Identify speaker by matching to existing characters only.
    All characters have been pre-extracted, so this is purely a matching problem.
    NO character creation - returns "unknown" if can't match.
    
    Args:
        dialogue_line: The dialogue line to identify speaker for
        context_before: Formatted string with context and dialogue tags
        context_after: Formatted string with following context
        character_map: Complete character database from Pass 1
        async_openai_client: OpenAI client
        model_name: Name of the model to use
    
    Returns:
        speaker_name (string) - canonical name from character_map or "unknown"
    """
    try:
        no_think_token = check_if_have_to_include_no_think_token()
        
        # Build list of valid speaker names (including narrator and unknown)
        valid_speakers = ["narrator", "unknown"] + [
            char_name for char_name in character_map.keys() if char_name != "narrator"
        ]
        
        # Format character map for LLM (exclude narrator)
        character_list = []
        for char_name, char_info in character_map.items():
            if char_name == "narrator":
                continue
            character_list.append(
                f"- {char_name}: {char_info.get('description', 'No description')} "
                f"(Age: {char_info.get('age', 'unknown')}, Gender: {char_info.get('gender', 'unknown')})"
            )
        character_context = "\n".join(character_list) if character_list else "No characters identified yet."
        
        # Create numbered list of EXACT valid names for emphasis
        exact_names_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(valid_speakers)])
        
        # Strengthened system prompt emphasizing EXACT name matching
        system_prompt = f"""{no_think_token}
You are an expert at matching dialogue to speakers. Your ONLY task is to determine which of the known characters is speaking the given dialogue.

ALL characters have been pre-extracted. Your job is PURELY MATCHING - DO NOT create new characters.

Known characters (COMPLETE LIST - all characters are here):
{character_context}

CRITICAL INSTRUCTIONS:

1. **EXACT NAME MATCHING - MANDATORY**:
   - You MUST return the EXACT canonical name EXACTLY as it appears in the valid names list below
   - DO NOT modify, shorten, lengthen, or change the name in ANY way
   - DO NOT use titles or variations - use the EXACT name from the list
   - WRONG: "vernon dursley" when list has "mr. dursley"
   - WRONG: "mr. dursley" when list has "vernon dursley"  
   - WRONG: "Harry" when list has "harry potter"
   - CORRECT: Copy the EXACT name from the valid list below

2. **VALID SPEAKER NAMES (COPY EXACTLY FROM THIS LIST)**:
{exact_names_list}

3. **Matching Process**:
   - FIRST: Check [CURRENT LINE WITH DIALOGUE TAGS] in context_before for dialogue tags
   - Dialogue tags show speaker hints: "said Harry", "Mr. Dursley replied", "she whispered"
   - Match the hint (name/title/pronoun) to a character description above
   - Find which character from the list best matches
   - Return that character's EXACT canonical name from the valid list (step 2)
   
4. **Name Variation Matching Examples**:
   - Dialogue tag: "Mr. Dursley said" | List contains "mr. dursley" -> Return "mr. dursley" (EXACT)
   - Dialogue tag: "Vernon said" | List contains "vernon dursley" -> Return "vernon dursley" (EXACT)
   - Dialogue tag: "Dumbledore said" | List contains "albus dumbledore" -> Return "albus dumbledore" (EXACT)
   - Dialogue tag: "said Harry" | List contains "harry potter" -> Return "harry potter" (EXACT)
   - Dialogue tag: "she replied" | Context shows Hermione | List has "hermione granger" -> Return "hermione granger" (EXACT)

5. **Special Cases**:
   - If the "dialogue" is actually narrative text (malformed quotes) -> Return "narrator"
   - If truly ambiguous with no matching clues -> Return "unknown"
   - NEVER invent new names - all valid names are in the list above

6. **Context Clues**:
   - Use previous dialogue marked [speaker_name]: to track conversation
   - Use pronouns (he/she/they) with character gender/age to match
   - Use character descriptions to identify references

VALIDATION REQUIREMENT:
Your response will be validated against the valid names list. If you return a name not in the list, it will cause an error.
You MUST copy the name EXACTLY character-by-character from the valid list."""
        
        # Create user prompt
        user_prompt = f"""Match this dialogue to a known character:

<context_before>
{context_before}
</context_before>

<dialogue_line>
{dialogue_line}
</dialogue_line>

<context_after>
{context_after}
</context_after>

CRITICAL: Return the EXACT canonical name from the valid names list in the system prompt. Copy it character-by-character without modification."""

        if is_chinese():
            system_prompt = build_zh_speaker_matching_prompt(
                no_think_token,
                character_context,
                exact_names_list,
            )
            user_prompt = f"""请将以下对话匹配到一个已知人物：

<context_before>
{context_before}
</context_before>

<dialogue_line>
{dialogue_line}
</dialogue_line>

<context_after>
{context_after}
</context_after>

重要：speaker 必须从系统提示词的有效说话人名单中逐字复制，不得改写或创建新名字。"""
        
        # Create dynamic Enum for speaker validation based on character_map
        # This creates a Pydantic enum that only accepts valid speaker names
        from enum import Enum
        SpeakerEnum = Enum('SpeakerEnum', {name.replace(' ', '_').replace('.', '_').replace('-', '_'): name for name in valid_speakers})
        
        # Pydantic model with constrained speaker field
        class SpeakerMatch(BaseModel):
            reasoning: str = Field(description="Brief explanation: 1) What dialogue tag/clue found, 2) Which character it matches from the list, 3) EXACT name being returned")
            speaker: SpeakerEnum = Field(description="EXACT canonical character name from valid speakers list - copy character-by-character without any modification")
        
        # Manual retry logic
        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            try:
                current_presence = PRESENCE_PENALTY + (attempt * 0.2)
                current_frequency = FREQUENCY_PENALTY + (attempt * 0.2)
                current_repeat = REPEAT_PENALTY + (attempt * 0.05)
                
                if attempt > 0:
                    print(f"🔄 Speaker Matching - Retry attempt {attempt + 1}/{max_retries}")
                
                provider = OpenAIProvider(openai_client=async_openai_client)
                model = OpenAIChatModel(model_name, provider=provider)
                agent = Agent(
                    model=model,
                    output_type=SpeakerMatch,
                    retries=3,
                    model_settings={
                        "temperature": TEMPERATURE,
                        "presence_penalty": current_presence,
                        "frequency_penalty": current_frequency,
                        "repeat_penalty": current_repeat
                    },
                    system_prompt=system_prompt
                )
                
                result = await agent.run(user_prompt)
                token_usage = result.usage()
                print(f"📊 Speaker Matching - Usage: {token_usage.input_tokens} input, {token_usage.output_tokens} output, {token_usage.total_tokens} total")
                
                break
                
            except Exception as e:
                if is_context_overflow_error(e):
                    if attempt < max_retries - 1:
                        print(f"⚠️  Context overflow detected. Retrying with higher penalties...")
                        continue
                    else:
                        print(f"❌ Context overflow persisted after {max_retries} retries.")
                        raise
                else:
                    raise
        
        if result is None:
            raise Exception("Failed to get result after retries")
    
        match: SpeakerMatch = result.output
        # Extract the actual speaker name from the enum (the enum value is the character name)
        speaker = match.speaker.value if hasattr(match.speaker, 'value') else str(match.speaker)
        
        # Logging
        print(f"🎯 Matched dialogue to: {speaker} | Reasoning: {match.reasoning[:100]}...")
        
        return speaker
    
    except Exception as e:
        print(f"Error matching speaker: {e}")
        traceback.print_exc()
        return "unknown"


async def llm_identify_characters_and_output_to_jsonl(
    text: str, 
    async_openai_client, 
    model_name: str, 
) -> Tuple[List[Dict], Set[str], Dict[str, Dict]]:
    """
    TWO-PASS APPROACH: Extract all characters first, then attribute dialogue.
    
    Pass 1: Extract ALL characters from entire text (including dialogue)
    Pass 2: Attribute speakers to dialogue (pure matching, no character creation)
    
    Args:
        text: The full book text
        async_openai_client: OpenAI client for LLM calls
        model_name: Name of the LLM model to use
    
    Returns:
        Tuple of (line_map, found_characters_set, character_map)
        - line_map: List of dicts with {"speaker": str, "line": str}
        - found_characters_set: Set of all character names found
        - character_map: Dict mapping character names to their demographic info
    """
    print("\n🚀 Starting TWO-PASS character identification...")
    
    # Preprocess text into simple sequence of items
    print("📝 Preprocessing text (splitting mixed lines)...")
    processed_items = preprocess_text_into_lines(text)
    total_items = len(processed_items)
    print(f"✅ Preprocessed into {total_items} items")
    
    # ==================== PASS 1: Extract ALL characters ====================
    print(f"\n{'='*80}")
    print("🔍 PASS 1: Extracting characters from ENTIRE text (narrator + dialogue)")
    print(f"{'='*80}")
    
    character_map = None
    async for update in extract_all_characters_from_full_text(
        processed_items,
        async_openai_client,
        model_name,
    ):
        yield update
        character_map = update
    
    print(f"\n✅ PASS 1 Complete!")
    print(f"📊 Extracted {len(character_map) - 1} characters (excluding narrator)")
    
    # Save intermediate character_gender_map after Pass 1
    print("💾 Saving character_gender_map.json after Pass 1...")
    character_gender_map_pass1 = {
        "legend": {
            "1": "completely masculine",
            "2": "mostly masculine",
            "3": "moderately masculine",
            "4": "slightly masculine",
            "5": "neutral/unknown",
            "6": "slightly feminine",
            "7": "moderately feminine",
            "8": "mostly feminine",
            "9": "almost completely feminine",
            "10": "completely feminine"
        },
        "scores": character_map
    }
    write_json_to_file(character_gender_map_pass1, "character_gender_map.json")
    print("✅ Character map saved after Pass 1")
    
    # ==================== PASS 2: Attribute speakers ====================
    print(f"\n{'='*80}")
    print("🎯 PASS 2: Attributing speakers to dialogue")
    print(f"{'='*80}")
    
    line_map = None
    async for update in attribute_speakers_to_dialogue(
        processed_items,
        character_map,
        async_openai_client,
        model_name
    ):
        yield update
        line_map = update
    
    print(f"\n✅ PASS 2 Complete!")
    print(f"📝 Attributed {sum(1 for item in line_map if item['speaker'] != 'narrator')} dialogue lines")
    
    # Build found_characters set (exclude special characters)
    found_characters = set(character_map.keys())
    found_characters.discard("narrator")
    found_characters.discard("unknown")
    
    # ==================== VALIDATION: Check for mismatches ====================
    print(f"\n{'='*80}")
    print("🔍 VALIDATION: Checking for speaker mismatches...")
    print(f"{'='*80}")
    
    # Extract all unique speakers from line_map
    speakers_in_output = set(item['speaker'] for item in line_map)
    speakers_in_character_map = set(character_map.keys())
    
    # Find speakers in output but not in character map
    invalid_speakers = speakers_in_output - speakers_in_character_map
    
    if invalid_speakers:
        print(f"❌ MISMATCH DETECTED!")
        print(f"   Speakers in output NOT in character map: {sorted(invalid_speakers)}")
        print(f"   Count: {len(invalid_speakers)}")
        
        # Show sample occurrences
        print(f"\n   Sample occurrences:")
        for speaker in sorted(invalid_speakers)[:5]:  # Show first 5
            sample_lines = [item['line'][:60] + "..." if len(item['line']) > 60 else item['line'] 
                          for item in line_map if item['speaker'] == speaker][:2]
            print(f"   - '{speaker}':")
            for line in sample_lines:
                print(f"     • {line}")
    else:
        print(f"✅ NO MISMATCHES FOUND!")
        print(f"   All speakers in output exist in character map.")

    # Write the processed lines to a JSONL file
    write_jsons_to_jsonl_file(line_map, "speaker_attributed_book.jsonl")
    
    print(f"\n📊 Statistics:")
    print(f"   - Speakers in character map: {len(speakers_in_character_map)}")
    print(f"   - Unique speakers in output: {len(speakers_in_output)}")
    print(f"   - Invalid speakers: {len(invalid_speakers)}")
    print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ TWO-PASS PROCESSING COMPLETE!")
    print(f"📊 Total characters: {len(found_characters)}")
    print(f"📝 Total items processed: {len(line_map)}")
    print(f"{'='*80}\n")
    
    yield line_map, found_characters, character_map


async def extract_all_characters_from_full_text(
    processed_items: List[Dict],
    async_openai_client,
    model_name: str,
) -> Dict[str, Dict]:
    """
    PASS 1: Extract all characters from the entire text (narrator + dialogue).
    Processes ALL text content to build a complete character database.
    
    Args:
        processed_items: List of preprocessed items from preprocess_text_into_lines()
        async_openai_client: OpenAI client
        model_name: Model name
    
    Returns:
        Complete character_map with all characters and their demographics
    """
    character_map: Dict[str, Dict] = {
        "narrator": {
            "name": "narrator",
            "age": "adult",
            "gender": "female",
            "gender_score": 0,
            "description": "The narrator of the story"
        },
        "unknown": {
            "name": "unknown",
            "age": "adult",
            "gender": "unknown",
            "gender_score": 5,
            "description": "Unknown or ambiguous speaker"
        }
    }
    
    batch: List[str] = []
    total_items = len(processed_items)
    
    for idx, item in enumerate(processed_items):
        progress_pct = int((idx + 1) * 100 / total_items)
        yield (f"Pass 1: Extracting Characters. Progress: {idx + 1}/{total_items} ({progress_pct}%)")
        print(f"Pass 1 Progress: {idx + 1}/{total_items} ({progress_pct}%)")
        
        content = item["content"]
        
        # Skip empty lines
        if not content.strip():
            continue

        # Add ALL content (both narrator and dialogue) to batch
        batch.append(content)
        
        # Process batch if it exceeds token limit
        batch_text = "\n".join(batch)
        current_batch_tokens = count_tokens(batch_text)
        
        if current_batch_tokens >= MAX_BATCH_TOKENS:
            updates, merges = await extract_character_info_from_batch(
                batch,
                character_map,
                async_openai_client,
                model_name
            )
            # Remove old names that were merged and track them
            for old_name, new_name in merges.items():
                if old_name in character_map:
                    del character_map[old_name]
            # Add/update characters
            character_map.update(updates)
            batch = []
    
    # Process any remaining batch
    if batch:
        updates, merges = await extract_character_info_from_batch(
            batch,
            character_map,
            async_openai_client,
            model_name
        )
        # Remove old names that were merged
        for old_name, new_name in merges.items():
            if old_name in character_map:
                del character_map[old_name]
        # Add/update characters
        character_map.update(updates)
    
    # Return character_map for downstream processing
    yield character_map


async def attribute_speakers_to_dialogue(
    processed_items: List[Dict],
    character_map: Dict[str, Dict],
    async_openai_client,
    model_name: str
) -> List[Dict]:
    """
    PASS 2: Attribute speakers to dialogue lines.
    Pure matching problem - all characters have been pre-extracted.
    
    Args:
        processed_items: List of preprocessed items
        character_map: Complete character database from Pass 1
        async_openai_client: OpenAI client
        model_name: Model name
    
    Returns:
        line_map with speaker attributions for all items
    """
    line_map: List[Dict] = []
    total_items = len(processed_items)
    
    for idx, item in enumerate(processed_items):
        progress_pct = int((idx + 1) * 100 / total_items)
        yield (f"Pass 2: Attributing Speakers. Progress: {idx + 1}/{total_items} ({progress_pct}%)")
        print(f"Pass 2 Progress: {idx + 1}/{total_items} ({progress_pct}%)")
        
        item_type = item["type"]
        content = item["content"]
        
        # Narrator items - no speaker attribution needed
        if item_type == "narrator" or not content.strip():
            line_map.append({"speaker": "narrator", "line": content})
            continue
        
        # Dialogue items - identify speaker
        context_before_text, context_after_text = get_context_for_item(
            processed_items,
            idx,
            line_map,
            MAX_CONTEXT_TOKENS_PER_DIRECTION
        )
        
        # Add full original line to context for dialogue tag extraction
        full_original_line = item["original_line"]
        enhanced_context_before = f"{context_before_text}\n\n[CURRENT LINE WITH DIALOGUE TAGS]: {full_original_line}"
        
        # Identify speaker (now purely matching - no character creation)
        speaker = await identify_speaker_for_dialogue_matching_only(
            content,  # dialogue fragment
            enhanced_context_before,
            context_after_text,
            character_map,
            async_openai_client,
            model_name
        )
        
        line_map.append({"speaker": speaker, "line": content})
    
    yield line_map
