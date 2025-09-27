import json
import os
import re
import argparse
import logging
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch
import wave
import struct
from pathlib import Path
import sys
import io
import csv

# --- Logging Setup ---
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Set lowest level to capture all messages

# Create file handler which logs even debug messages
fh = logging.FileHandler('match_evo.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(message)s')
fh.setFormatter(file_formatter)
ch.setFormatter(console_formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# --- 配置 ---
# 新版本场景数据文件
NEW_SCENA_FILE = 'scena_data_jp_Command.json'
# 新版本场景中文翻译
NEW_SCENA_SC_TRANSLATION = 'scena_data_sc_Command.json'
# 旧版本语音数据文件
OLD_VOICE_FILE = 'voice_data.json'
# 旧版本脚本数据文件
OLD_SCRIPT_FILE = 'script_data.json'
# 输出文件：成功匹配的数据
MERGED_OUTPUT_FILE = 'merged_voice_data.json'
# 输出文件：未成功匹配的数据
UNMATCHED_OUTPUT_FILE = 'unmatched_voice_data.json'
# 输出文件：跳过匹配的数据
SKIPPED_OUTPUT_FILE = 'skipped_voice_data.json'
# 输出文件：匹配结果CSV
MATCH_RESULT_CSV = 'match_result.csv'
# 新编号开始
NEW_ID_START = 50001
# Normalize 检索使用的最小非标点符号字符数
NORMALIZE_SEARCH_THRESHOLD = 4
# Exact 检索使用的最短字符数
EXACT_SEARCH_THRESHOLD = 8
# Vector Search without context 使用的最短字符数
VECTOR_SEARCH_WITHOUT_CONTEXT_THRESHOLD = 8
# Vector Search without context (vector_single) 使用的相似度阈值
VECTOR_SEARCH_WITHOUT_CONTEXT_SIMILARITY_THRESHOLD = 0.99

def normalize_text(text):
    """移除文本中的所有标点符号和空格，用于匹配。"""
    # 此正则表达式应能处理日文和英文标点
    return re.sub(r'[\s\W]', '', text)

def classify_voice_file(filename):
    """
    根据文件名对语音文件进行分类。

    Args:
        filename (str): 语音文件的名称。

    Returns:
        dict: 包含分类信息的字典。
              例如 {'type': 'character_voice', 'character_id': '001', 'category': 'main', 'number': '0001'}
              或 {'type': 'sound_effect', 'description': 'cat_02'}
    """
    # 角色语音: v<角色ID>_<类型>_<序号>.wav (e.g., v001_00_0001.wav, v327_gs_0002.wav)
    match = re.match(r'^v(\d{3})_(\w{2})_(\d{4}[br]?)\.wav$', filename)
    if match:
        char_id, category_code, number = match.groups()

        category = 'unknown'
        if category_code.isdigit():
            category = 'main'
        else:
            category_map = {
                'av': 'active_voice',
                'bv': 'battle_voice'
            }
            category = category_map.get(category_code, 'unknown')

        return {
            'type': 'character_voice',
            'character_id': char_id,
            'category': category,
            'number': number
        }

    # 战斗/系统语音: v<角色ID>_<类型><序号>.wav (e.g., v001_b0001.wav, v001_s0001.wav, v001_b0118b.wav)
    match = re.match(r'^v(\d{3})_([bs])(\d{4}[bcr]?)\.wav$', filename)
    if match:
        char_id, category_code, number = match.groups()
        category_map = {
            'b': 'battle',
            's': 'system'
        }
        return {
            'type': 'character_voice',
            'character_id': char_id,
            'category': category_map.get(category_code, 'unknown'),
            'number': number
        }

    # 音效: v_se_<描述>.wav (e.g., v_se_cat_02.wav)
    match = re.match(r'^v_se_(.*)\.wav$', filename)
    if match:
        description = match.group(1)
        return {
            'type': 'sound_effect',
            'description': description
        }

    return {'type': 'unknown', 'filename': filename}

def find_best_match(new_entry, old_data_map, old_data_normalized_map, old_script_list, model, old_embeddings, old_embeddings_no_context, args, used_old_voice_ids, methods):
    """执行指定方法的匹配策略来查找最佳匹配。"""
    new_text = new_entry['text']

    # 1. 精确匹配
    if 'exact' in methods and new_text in old_data_map:
        if len(new_text) >= EXACT_SEARCH_THRESHOLD: 
            candidates = old_data_map[new_text]
            candidate_embeddings = model.encode([f"{c.get('context_prev', '')} {c['text']} {c.get('context_next', '')}" for c in candidates], convert_to_tensor=True)
            query_embedding = model.encode(f"{new_entry.get('context_prev', '')} {new_text} {new_entry.get('context_next', '')}", convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, candidate_embeddings, top_k=1)
            return candidates[hits[0][0]['corpus_id']], 'exact'

    # 2. 移除标点后匹配
    if 'normalized' in methods and (normalized_new_text := normalize_text(new_text)) and normalized_new_text in old_data_normalized_map:
        if len(normalized_new_text) >= NORMALIZE_SEARCH_THRESHOLD:
            candidates = old_data_normalized_map[normalized_new_text]
            candidate_embeddings = model.encode([f"{c.get('context_prev', '')} {c['text']} {c.get('context_next', '')}" for c in candidates], convert_to_tensor=True)
            query_embedding = model.encode(f"{new_entry.get('context_prev', '')} {new_text} {new_entry.get('context_next', '')}", convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, candidate_embeddings, top_k=1)
            return candidates[hits[0][0]['corpus_id']], 'normalized'

    # 3. 向量相似度匹配
    if 'vector' in methods and not args.no_similarity_search:
        contextual_new_text = f"{new_entry.get('context_prev', '')} {new_text} {new_entry.get('context_next', '')}".strip()
        query_embedding = model.encode(contextual_new_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, old_embeddings, top_k=1)
        if hits and hits[0][0]['score'] > args.similarity_threshold:
            best_match_candidate = old_script_list[hits[0][0]['corpus_id']]
            score = hits[0][0]['score']
            match_type = f'vector_search ({score:.2f})'
            # check vector similarity of old text and matched text, in ignorance of context
            best_match_text = best_match_candidate['text']
            new_text = new_entry['text']
            best_match_text_embedding = model.encode(best_match_text, convert_to_tensor=True)
            new_text_embedding = model.encode(new_text, convert_to_tensor=True)
            # calculate vector similarity, don't use semantic_search
            text_similarity = util.cos_sim(new_text_embedding, best_match_text_embedding)[0][0].item()
            if text_similarity >= args.similarity_threshold:
                return best_match_candidate, match_type
    
    # 4. 在无上下文的情况下使用向量相似度匹配
    if 'vector_single' in methods and not args.no_similarity_search and len(new_text) >= VECTOR_SEARCH_WITHOUT_CONTEXT_THRESHOLD:
        query_embedding = model.encode(new_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, old_embeddings_no_context, top_k=1)
        if hits and hits[0][0]['score'] > VECTOR_SEARCH_WITHOUT_CONTEXT_SIMILARITY_THRESHOLD:
            best_match_candidate = old_script_list[hits[0][0]['corpus_id']]
            match_type = f'vector_single ({hits[0][0]["score"]:.2f})'
            return best_match_candidate, match_type

    return None, None

def blockwise_match(scripts, voice_table, old_voice_id_to_entry_map):
    """按照3个为一组进行匹配，之后将匹配结果之间的空隙使用边界的匹配结果作为提示再次匹配，输出匹配的结果"""
    scripts = [s for s in scripts if s.get('text')]
    voice_table = [v for v in voice_table if v.get('text')]
    script_slide_iterators = [scripts[i:] for i in range(3)]
    voice_table_slide_iterators = [voice_table[i:] for i in range(3)]
    script_id_map = {script['script_id']: script for script in scripts}
    voice_table_id_map = {voice['id']: voice for voice in voice_table}
    script_context_map = {script_slide[0]['text'] + "||" + script_slide[1]['text'] + "||" + script_slide[2]['text']: script_slide for script_slide in zip(*script_slide_iterators)}
    voice_table_context_map = {voice_slide[0]['text'] + "||" + voice_slide[1]['text'] + "||" + voice_slide[2]['text']: voice_slide for voice_slide in zip(*voice_table_slide_iterators)}
    voice_table_context_match = {}
    for voice_context, voice_slide in voice_table_context_map.items():
        if voice_context in script_context_map:
            matched_pairs = zip(voice_slide, script_context_map[voice_context])
            for voice, script in matched_pairs:
                voice_table_context_match[voice['id']] = script['script_id']
    
    voice_table_context_match_stage1 = [(voice['id'], voice_table_context_match.get(voice['id'],None)) for voice in voice_table]
    for voice_id, script_id in voice_table_context_match_stage1:
        if script_id is not None:
            logger.info(f"{voice_id}: {script_id} | {script_id_map[script_id]['voice_id']}")
        else:
            logger.info(f"{voice_id}: None")
    
    def find_none_blocks(matched_items):
        """Find continuous subranges of None values in a list of matched items."""
        match_margins = []
        in_none_block = False
        start_index = -1
        
        for i, (_, script_id) in enumerate(matched_items):
            if script_id is None and not in_none_block:
                in_none_block = True
                start_index = i
            elif script_id is not None and in_none_block:
                in_none_block = False
                match_margins.append((start_index, i - 1))

        if in_none_block:
            match_margins.append((start_index, len(matched_items) - 1))

        # For each margin, find the hints
        margin_hints = []
        for start, end in match_margins:
            hint_before = matched_items[start - 1] if start > 0 else None
            hint_after = matched_items[end + 1] if end < len(matched_items) - 1 else None
            if hint_before is not None and hint_after is not None:
                margin_hints.append({
                'range': (start, end),
                'hint_before': hint_before,
                'hint_after': hint_after
            })
            
        return match_margins, margin_hints

    context_match_margins_stage1, script_margin_hints = find_none_blocks(voice_table_context_match_stage1)


    script_margin_hint_match = {}
    for margin_hint in script_margin_hints :
        start_idx, end_idx = margin_hint['range']
        hint_before, hint_after = margin_hint['hint_before'], margin_hint['hint_after']
        if end_idx - start_idx + 1 == (hint_after[1] - 1) - (hint_before[1] + 1) + 1:
            logger.info(f"Found continuous margin: {margin_hint}")
            for idx in range(start_idx, end_idx + 1):
                predicate_script_id = hint_before[1]+1 + (idx - start_idx)
                if predicate_script_id not in script_id_map:
                    logger.info(f"Mapping {voice_table_context_match_stage1[idx][0]} to None | Skipped")
                    continue
                logger.info(f"Mapping {voice_table_context_match_stage1[idx][0]} to {predicate_script_id} | {script_id_map[predicate_script_id]['voice_id']}")
                script_margin_hint_match[voice_table_context_match_stage1[idx][0]] = predicate_script_id

    voice_table_context_match.update(script_margin_hint_match)
    voice_table_context_match_stage2 = [(voice['id'], voice_table_context_match.get(voice['id'], None)) for voice in voice_table]
    
    context_match_margins_stage2, voice_margin_hints= find_none_blocks(voice_table_context_match_stage2)

    def get_old_voice_scene_order(old_voice_id):
        return int(old_voice_id[3:10])

    old_voice_scene_order_to_entry_map = {get_old_voice_scene_order(old_voice_id): old_voice_entry for old_voice_id, old_voice_entry in old_voice_id_to_entry_map.items()}

    voice_margin_hint_match = {}
    for margin_hint in voice_margin_hints :
        start_match_idx, end_match_idx = margin_hint['range']
        hint_before, hint_after = margin_hint['hint_before'], margin_hint['hint_after']
        hint_before_old_voice_id, hint_after_old_voice_id = script_id_map[hint_before[1]].get('voice_id'), script_id_map[hint_after[1]].get('voice_id')
        hint_before_old_voice_scene_order = get_old_voice_scene_order(hint_before_old_voice_id)
        hint_after_old_voice_scene_order = get_old_voice_scene_order(hint_after_old_voice_id)
        if (hint_after_old_voice_scene_order - 1) - (hint_before_old_voice_scene_order + 1) + 1 == end_match_idx - start_match_idx + 1:
            is_all_has_voice = True
            for idx in range(start_match_idx, end_match_idx + 1):
                predicate_voice_id = hint_before_old_voice_scene_order + 1 + idx - start_match_idx
                if old_voice_scene_order_to_entry_map.get(predicate_voice_id) is None:
                    is_all_has_voice = False
                    break
            if is_all_has_voice:
                logger.info(f"Found continuous margin: {margin_hint}")
                for idx in range(start_match_idx, end_match_idx + 1):
                    predicate_voice_id = hint_before_old_voice_scene_order + 1 + idx - start_match_idx
                    logger.info(f"Mapping {voice_table_context_match_stage2[idx][0]} to {old_voice_scene_order_to_entry_map[predicate_voice_id]['script_id']} | {old_voice_scene_order_to_entry_map[predicate_voice_id]['voice_id']}")
                    voice_table_context_match[voice_table_context_match_stage2[idx][0]] = old_voice_scene_order_to_entry_map[predicate_voice_id]['script_id']
                
    voice_table_context_match.update(voice_margin_hint_match)    
    voice_table_context_match_to_old_voice_id = {id: old_voice_id_to_entry_map[script_id_map[script_id]['voice_id']] for id, script_id in voice_table_context_match.items()}

    return voice_table_context_match_to_old_voice_id

def main():
    """主函数，执行匹配和生成文件。"""
    parser = argparse.ArgumentParser(description='匹配新旧语音数据。')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='输出详细的日志信息，包括跳过的条目。'
    )
    parser.add_argument('--no-similarity-search', action='store_true', help='禁用向量相似度搜索')
    parser.add_argument('--similarity-threshold', type=float, default=0.85, help='设置向量相似度搜索的阈值 (默认: 0.85)')
    args = parser.parse_args()

    try:
        with open(NEW_SCENA_FILE, 'r', encoding='utf-8') as f:
            new_scena_data = json.load(f)
        with open(OLD_VOICE_FILE, 'r', encoding='utf-8') as f:
            old_data_list = json.load(f)
        with open(OLD_SCRIPT_FILE, 'r', encoding='utf-8') as f:
            old_script_list = json.load(f)
        with open(NEW_SCENA_SC_TRANSLATION, 'r', encoding='utf-8') as f:
            new_scena_translation = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"错误：找不到文件 {e.filename}")
        return
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"错误：解析JSON文件或找不到键时出错: {e}")
        return

    new_data = []
    # 为新场景文件添加序号，生成new_data
    for i, entry in enumerate(new_scena_data):
        new_entry = {}
        new_entry['id'] = NEW_ID_START + i
        new_entry['text'] = entry['args'][-1]
        if len(entry['args']) >= 3:
            if entry['args'][-3] == 11 and isinstance(entry['args'][-2],int):
                new_entry['remake_voice_id'] = entry['args'][-2]
        if len(entry['args']) >= 2:
            if entry['args'][-2] == 11 and isinstance(entry['args'][-1],int):
                new_entry['remake_voice_id'] = entry['args'][-1]
                new_entry['text'] = ""
        new_entry['filebase'] = Path(entry['file']).stem
        new_entry['lineno'] = entry['line']
        new_entry['lineno_corr'] = entry.get('line_corr')
        new_data.append(new_entry)

    new_translation = []
    for i, entry in enumerate(new_scena_translation):
        new_entry = {}
        new_entry['id'] = NEW_ID_START + i
        new_entry['text'] = entry['args'][-1]
        if len(entry['args']) >= 2:
            if entry['args'][-2] == 11 and isinstance(entry['args'][-1],int):
                new_entry['text'] = ""
        new_entry['filebase'] = Path(entry['file']).stem
        new_entry['lineno'] = entry['line']
        new_entry['lineno_corr'] = entry.get('line_corr')
        new_translation.append(new_entry)

    new_translation_filemap = {entry['id']: entry for entry in new_translation}
    # 为新语音增加翻译文本
    for entry in new_data:
        entry['translation'] = new_translation_filemap.get(entry['id'], {}).get('text', '')


    # 为新语音添加上下文
    for i, entry in enumerate(new_data):
        # 添加上一句上下文
        if i > 0:
            entry['context_prev'] = new_data[i-1].get('text', '')
        else:
            entry['context_prev'] = ""
            
        # 添加下一句上下文
        if i < len(new_data) - 1:
            entry['context_next'] = new_data[i+1].get('text', '')
        else:
            entry['context_next'] = ""
    logger.info("上下文添加完成。")
        

    # 为旧语音数据添加上下文
    logger.info("正在为旧语音数据添加上下文...")
    # 从 voice_id 解析场景信息并排序
    for entry in old_data_list:
        voice_id = entry.get('voice_id', '')
        if isinstance(voice_id, str) and len(voice_id) >= 10:
            entry['scene_id'] = voice_id[3:6]
            entry['scene_seq_id'] = int(voice_id[6:10])
        else:
            # 为不符合格式的ID设置默认值以便排序
            entry['scene_id'] = ""
            entry['scene_seq_id'] = -1

    old_data_list.sort(key=lambda x: (x.get('scene_id', ''), x.get('scene_seq_id', '')))
    for i, entry in enumerate(old_data_list):
        # 检查是否在同一场景内
        is_same_scene_prev = (i > 0 and 
                              old_data_list[i-1].get('scene_id') == entry.get('scene_id'))
        
        is_same_scene_next = (i < len(old_data_list) - 1 and 
                              old_data_list[i+1].get('scene_id') == entry.get('scene_id'))

        # 添加上一句上下文
        if is_same_scene_prev:
            entry['context_prev'] = old_data_list[i-1].get('text', '')
        else:
            entry['context_prev'] = ""
        
        # 添加下一句上下文
        if is_same_scene_next:
            entry['context_next'] = old_data_list[i+1].get('text', '')
        else:
            entry['context_next'] = ""
    logger.info("旧数据上下文添加完成。")

    if not args.no_similarity_search:
        # 加载预训练的 sentence-transformer 模型
        # 'paraphrase-multilingual-MiniLM-L12-v2' 是一个性能优秀的多语言模型
        logger.info("正在加载文本向量化模型...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("模型加载完成。")

        # 为旧数据创建向量嵌入
        logger.info("正在为旧脚本数据创建上下文向量嵌入...")
        old_contextual_texts = [
            f"{entry.get('context_prev', '')} {entry.get('text', '')} {entry.get('context_next', '')}".strip()
            for entry in old_script_list
        ]
        old_embeddings = model.encode(old_contextual_texts, convert_to_tensor=True)
        old_embeddings_no_context = model.encode([entry['text'] for entry in old_script_list], convert_to_tensor=True)
        logger.info("向量嵌入创建完成。")
    else:
        logger.info("跳过向量嵌入创建，因为 --no-similarity-search 被设置。")
        model = None
        old_embeddings = None

    # 为旧数据创建快速查找映射（一个文本可能对应多个语音）
    old_data_map = defaultdict(list)
    old_data_normalized_map = defaultdict(list)
    for entry in old_data_list:
        if text := entry.get('text'):
            # 精确匹配映射
            old_data_map[text].append(entry)
            
            # 标准化文本映射
            normalized_text = normalize_text(text)
            if normalized_text:
                old_data_normalized_map[normalized_text].append(entry)

    # 为旧脚本数据创建快速查找映射
    old_script_map = defaultdict(list)
    for entry in old_script_list:
        if text := entry.get('text'):
            old_script_map[text].append(entry)

    # 创建 voice_id 到 old_data_list 条目的映射
    old_voice_id_to_entry_map = {e['voice_id']: e for e in old_data_list}

    # 对候选项列表进行排序，确保优先匹配文件名靠前的语音
    logger.info("正在对具有相同文本的候选项进行排序...")
    for text in old_data_map:
        old_data_map[text].sort(key=lambda e: e.get('voice_id', ''))
    for text in old_data_normalized_map:
        old_data_normalized_map[text].sort(key=lambda e: e.get('voice_id', ''))
    for text in old_script_map:
        old_script_map[text].sort(key=lambda e: e.get('script_id', 0))
    logger.info("排序完成。")

    matched_data = []
    unmatched_data = []
    skipped_data = []
    used_old_voice_ids = set() # 用于跟踪已匹配的旧语音ID
    
    total_count = len(new_data)
    processed_count = len(new_data)
    vector_search_success_count = 0

    # 筛选出需要处理的条目
    entries_to_process = new_data
    logger.info(f"开始处理 {processed_count} 条符合条件的语音数据...")

    # --- Pass 1: Exact and Normalized Matching ---
    logger.info("\n--- 第一遍: 执行精确匹配和标准化匹配 ---")
    remaining_entries_pass2 = []
    pass1_success_count = 0
    blockwise_match_result = blockwise_match(old_script_list, entries_to_process, old_voice_id_to_entry_map)
    for new_entry in entries_to_process:
        match_type = "blockwise"
        best_match = blockwise_match_result.get(new_entry.get('id'))

        if best_match:
            pass1_success_count += 1
            used_old_voice_ids.add(best_match['voice_id'])
            matched_data.append({
                'new_voice_id': new_entry.get('id'),
                'new_script_file': new_entry.get('filebase'),
                'new_script_lineno': new_entry.get('lineno'),
                'new_add_struct_lineno': new_entry.get('line_corr'),
                'new_text': new_entry['text'],
                'remake_voice_id': new_entry.get('remake_voice_id'),
                'new_translation': new_entry.get('translation'),
                'old_voice_id': best_match.get('voice_id'),
                'old_script_id': best_match.get('script_id'),
                'old_scene_id': best_match.get('scene_id'),
                'old_scene_seq_id': best_match.get('scene_seq_id'),
                'old_text': best_match.get('text'),
                'character_id': best_match.get('character_id'),
                'source_file': best_match.get('source_file'),
                'match_type': match_type,
                'classification': {'category': 'evo'}
            })
        else:
            remaining_entries_pass2.append(new_entry)
    logger.info(f"第一遍完成: 成功匹配 {pass1_success_count} 条。")

    reverted_count = 0
    remaining_entries_pass2.sort(key=lambda x: x['id'])

    # --- Pass 2: Contextual Matching for Ambiguous Entries ---
    logger.info("\n--- 第二遍: 对剩余条目中存在歧义的部分执行上下文精确匹配 ---")
    pass2_success_count = 0
    remaining_entries_pass3 = []  # Entries that will go to vector search
    for new_entry in reversed(remaining_entries_pass2):
        new_text = new_entry.get('text', '')
        
        # Find potential candidates from script data for contextual matching
        candidates = old_script_map.get(new_text, [])
        
        # Only perform context match if there's ambiguity (multiple candidates)
        if new_text and len(candidates) > 1:
            found_context_match = False
            for candidate in candidates:

                # Triplet check: current text (already matches), previous, and next context
                if new_entry.get('context_prev', '') == candidate.get('context_prev', '') and \
                   new_entry.get('context_next', '') == candidate.get('context_next', ''):
                    
                    pass2_success_count += 1
                    logger.debug(f"  - 上下文匹配成功: New ID {new_entry['id']}")
                    logger.debug(f"    - New Context: ['{new_entry.get('context_prev', '')}', '{new_entry.get('text', '')}', '{new_entry.get('context_next', '')}']")
                    logger.debug(f"    - Old Context: ['{candidate.get('context_prev', '')}', '{candidate.get('text', '')}', '{candidate.get('context_next', '')}']")

                    used_old_voice_ids.add(candidate['voice_id'])
                    matched_data.append({
                        'new_voice_id': new_entry.get('id'),
                        'new_script_file': new_entry.get('filebase'),
                        'new_script_lineno': new_entry.get('lineno'),
                        'new_add_struct_lineno': new_entry.get('line_corr'),
                        'new_text': new_entry['text'],
                        'remake_voice_id': new_entry.get('remake_voice_id'),
                        'new_translation': new_entry.get('translation'),
                        'old_voice_id': candidate.get('voice_id'),
                        'old_script_id': candidate.get('script_id'),
                        'old_scene_id': candidate.get('scene_id'),
                        'old_scene_seq_id': candidate.get('scene_seq_id'),
                        'old_text': candidate.get('text'),
                        'character_id': candidate.get('character_id'),
                        'source_file': candidate.get('source_file'),
                        'match_type': 'context',
                        'classification': {'category': 'evo'}
                    })
                    found_context_match = True
                    break # Found a match, no need to check other candidates
            
            if not found_context_match:
                remaining_entries_pass3.append(new_entry)
        else:
            # If no ambiguity, pass to the next stage
            remaining_entries_pass3.append(new_entry)
    logger.info(f"第二遍完成: 成功匹配 {pass2_success_count} 条。")

    # --- Pass 3: Vector Similarity Matching ---
    logger.info("\n--- 第三遍: 对剩余条目执行向量相似度匹配 ---")
    pass3_success_count = 0
    for new_entry in remaining_entries_pass3:
        best_match, match_type = find_best_match(new_entry, old_data_map, old_data_normalized_map, old_script_list, model, old_embeddings, old_embeddings_no_context, args, used_old_voice_ids, methods=['exact', 'normalized', 'vector', 'vector_single'])

        if best_match:
            pass3_success_count += 1
            if match_type.startswith('vector_search') or match_type.startswith('vector_single'):
                vector_search_success_count += 1
                logger.debug(f"  - 向量相似度匹配成功: New ID {new_entry['id']} {match_type[:13]} {match_type[13:]}")
            else:
                logger.debug(f"  - 无上下文精确匹配成功: New ID {new_entry['id']} {match_type}")
            logger.debug(f"    - New Context: [{new_entry['context_prev']} {new_entry['text']} {new_entry['context_next']}]")
            logger.debug(f"    - Old Context: [{best_match['context_prev']} {best_match['text']} {best_match['context_next']}]")

            matched_data.append({
                'new_voice_id': new_entry.get('id'),
                'new_script_file': new_entry.get('filebase'),
                'new_script_lineno': new_entry.get('lineno'),
                'new_add_struct_lineno': new_entry.get('line_corr'),
                'new_text': new_entry['text'],
                'remake_voice_id': new_entry.get('remake_voice_id'),
                'new_translation': new_entry.get('translation'),
                'old_voice_id': best_match.get('voice_id'),
                'old_script_id': best_match.get('script_id'),
                'old_scene_id': best_match.get('scene_id'),
                'old_scene_seq_id': best_match.get('scene_seq_id'),
                'old_text': best_match.get('text'),
                'character_id': best_match.get('character_id'),
                'source_file': best_match.get('source_file'),
                'match_type': match_type,
                'classification': {'category':'evo'}
            })
        else:
            unmatched_data.append({
                'new_voice_id': new_entry.get('id'),
                'new_script_file': new_entry.get('filebase'),
                'new_script_lineno': new_entry.get('lineno'),
                'new_add_struct_lineno': new_entry.get('lineno_corr'),
                'text': new_entry['text'],
                'remake_voice_id': new_entry.get('remake_voice_id'),
                'new_translation': new_entry.get('translation'),
                'classification': {'category':'evo'}
            })
    logger.info(f"第三遍完成: 成功匹配 {pass3_success_count} 条。")

    # 最终成功数就是 matched_data 列表的长度
    success_count = len(matched_data)

    # 在写入前按 new_voice_id 排序
    matched_data.sort(key=lambda x: x['new_voice_id'])

    # 写入输出文件
    with open(MERGED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, ensure_ascii=False, indent=4)
    
    with open(UNMATCHED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unmatched_data, f, ensure_ascii=False, indent=4)
    
    with open(SKIPPED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(skipped_data, f, ensure_ascii=False, indent=4)

    matched_data_new_voice_id_map = {entry['new_voice_id']: entry for entry in matched_data}
    unmatched_data_new_voice_id_map = {entry['new_voice_id']: entry for entry in unmatched_data}
    skipped_data_new_voice_id_map = {entry['new_voice_id']: entry for entry in skipped_data}
    with open(MATCH_RESULT_CSV, 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(['RemakeVoiceID', 'RemakeScenaScriptFilename', 'RemakeScenaScriptLineno', 'RemakeScenaScriptAddStructLineno', 'OldScriptId', 'OldVoiceFilename', 'MatchType', 'RemakeVoiceCategory','RemakeVoiceTranslation', 'RemakeVoiceText', 'OldVoiceText'])
        rows_to_write = []
        for new_voice_entry in new_data:
            matched_entry = matched_data_new_voice_id_map.get(new_voice_entry['id'])
            if matched_entry:
                rows_to_write.append([
                    matched_entry['remake_voice_id'] or matched_entry['new_voice_id'],
                    matched_entry['new_script_file'],
                    matched_entry['new_script_lineno'],
                    matched_entry['new_add_struct_lineno'],
                    matched_entry['old_script_id'],
                    "ch" + matched_entry['old_voice_id'][:-1],
                    matched_entry['match_type'],
                    matched_entry['classification']['category'],
                    matched_entry['new_translation'],
                    matched_entry['new_text'],
                    matched_entry['old_text']
                ])
            else  :
                unmatched_entry = unmatched_data_new_voice_id_map.get(new_voice_entry['id'])
                if unmatched_entry:
                    rows_to_write.append([
                        unmatched_entry['remake_voice_id'] or unmatched_entry['new_voice_id'],
                        unmatched_entry['new_script_file'],
                        unmatched_entry['new_script_lineno'],
                        unmatched_entry['new_add_struct_lineno'],
                        '',
                        '',
                        'unmatched',
                        unmatched_entry['classification']['category'],
                        unmatched_entry['new_translation'],
                        unmatched_entry['text'],
                        ''
                    ])
                else:
                    skipped_entry = skipped_data_new_voice_id_map.get(new_voice_entry['id'])
                    rows_to_write.append([
                        skipped_entry['remake_voice_id'] or skipped_entry['new_voice_id'],
                        skipped_entry['new_script_file'],
                        skipped_entry['new_script_lineno'],
                        skipped_entry['new_add_struct_lineno'],
                        '',
                        '',
                        'skipped',
                        skipped_entry['classification'].get('category', ''),
                        skipped_entry['new_translation'],
                        skipped_entry['text'],
                        ''
                    ])
        writer.writerows(rows_to_write)

    # --- 打印统计结果 ---
    logger.info("\n--- 匹配完成 ---")
    logger.info(f"总计 (输入文件): {total_count}")
    logger.info(f"处理 (符合条件): {processed_count}")
    logger.info(f"成功: {success_count} (其中向量搜索: {vector_search_success_count})")
    logger.info(f"失败: {processed_count - success_count}")
    logger.info(f"成功匹配的数据已保存到: {MERGED_OUTPUT_FILE}")
    logger.info(f"未匹配的数据已保存到: {UNMATCHED_OUTPUT_FILE}")
    logger.info(f"跳过匹配的数据已保存到: {SKIPPED_OUTPUT_FILE}")
    logger.info(f"匹配结果已保存到: {MATCH_RESULT_CSV}")

if __name__ == '__main__':
    main()
