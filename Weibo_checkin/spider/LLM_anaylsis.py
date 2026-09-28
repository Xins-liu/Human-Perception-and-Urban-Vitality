import asyncio
import json
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
from openai import AsyncOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DeepSeekConfig:
    """DeepSeek API"""
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_concurrent: int = 5
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1
    enable_cache: bool = True
    cache_file: str = "deepseek_cache.json"
    cache_ttl: Optional[int] = None


@dataclass
class SentimentConfig:
    """Sentiment analysis"""
    input_file: str
    output_file: str
    text_column: str
    max_rows: Optional[int] = None
    batch_size: int = 20
    resume_file: str = "progress_checkpoint.jsonl"
    use_strict_mode: bool = True
    binary_classification: bool = True  # True-2，False-3


class OptimizedUrbanAnalyzer:
    """main"""

    def __init__(self, api_config: DeepSeekConfig, task_config: SentimentConfig):
        self.api_config = api_config
        self.task_config = task_config
        self.semaphore = asyncio.Semaphore(api_config.max_concurrent)
        self.client = AsyncOpenAI(api_key=api_config.api_key, base_url=api_config.base_url, timeout=api_config.timeout)
        self._cache = {}
        self._cache_loaded = False

        #


        # prompt
        self.system_prompt = f"""你是城市空间感知分析与社交媒体情感分析专家。请分析评论中的情感倾向和城市空间要素。

核心要求：
1. 情感评分0-1：充分考虑文本背景、用语与表情包元素，情感越积极分数越接近1，越消极越接近0，中性为0.5。
2. 情感分类：充分理解文本，进行深度情感分析，将文本进行情感三分类。
3. 城市空间要素：只提取与城市空间直接相关的要素（设施、环境、体验等）。
4. mention字段：简短概括（5-10字），不要复制原文。
5. 若无相关空间要素，返回空数组[]。
请严格使用工具，只返回JSON。"""

        # design
        self.tool = {
            "type": "function",
            "function": {
                "name": "analyze_urban_perception",
                "description": "分析城市空间感知和情感倾向，充分考虑表情包与网络用语",
                "strict": task_config.use_strict_mode,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "sentiment_category": {"type": "string", "enum": ["positive", "negative","neutral"]},
                        "urban_aspects": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aspect": {"type": "string"},
                                    "sentiment": {"type": "string", "enum": ["positive", "negative","neutral"]},
                                    "mention": {"type": "string"}
                                },
                                "required": ["aspect", "sentiment", "mention"]
                            }
                        }
                    },
                    "required": ["sentiment_score", "sentiment_category", "urban_aspects"]
                }
            }
        }

    def _get_cache_key(self, text: str) -> str:
        """"""
        return hashlib.md5(f"{self.system_prompt}:{text}".encode('utf-8')).hexdigest()

    def _load_cache(self):
        """"""
        if not self.api_config.enable_cache or self._cache_loaded:
            return
        cache_path = Path(self.api_config.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                current_time = time.time()
                self._cache = {k: v for k, v in cache_data.items()
                               if not self.api_config.cache_ttl or (
                                           current_time - v.get('timestamp', 0)) <= self.api_config.cache_ttl}
                logger.info(f": {len(self._cache)} ")
            except Exception as e:
                logger.warning(f"error: {e}")
                self._cache = {}
        self._cache_loaded = True

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """"""
        if not self.api_config.enable_cache:
            return None
        self._load_cache()
        return self._cache.get(cache_key, {}).get('result') if cache_key in self._cache else None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """"""
        if not self.api_config.enable_cache:
            return
        self._cache[cache_key] = {'result': result, 'timestamp': time.time()}

    def save_cache(self):
        """"""
        if not self.api_config.enable_cache or not self._cache:
            return
        try:
            cache_path = Path(self.api_config.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"save error: {e}")

    def load_data(self) -> pd.DataFrame:
        """data"""
        file_path = Path(self.task_config.input_file)
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_csv(file_path)
            logger.info(f"s: {len(df)} ")
            if self.task_config.text_column not in df.columns:
                raise ValueError(f" '{self.task_config.text_column}' not exit")
            if self.task_config.max_rows and self.task_config.max_rows < len(df):
                df = df.head(self.task_config.max_rows)
                logger.info(f": {self.task_config.max_rows} ")
            return df
        except Exception as e:
            logger.error(f"error: {e}")
            raise

    def load_processed_indices(self) -> set:
        """"""
        processed_indices = set()
        checkpoint_path = Path(self.task_config.resume_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            processed_indices.add(data.get('row_index', -1))
                logger.info(f": {len(processed_indices)} ")
            except Exception as e:
                logger.warning(f": {e}")
        return processed_indices

    def save_checkpoint(self, row_index: int, result: Dict):
        """"""
        try:
            with open(self.task_config.resume_file, 'a', encoding='utf-8') as f:
                checkpoint_data = {'row_index': row_index, 'timestamp': time.time(), 'result': result}
                f.write(json.dumps(checkpoint_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f": {e}")

    async def analyze_single(self, text: str, row_index: int) -> Dict[str, Any]:
        """"""
        cache_key = self._get_cache_key(text)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f" {row_index}: ")
            return {"row_index": row_index, "success": True, "result": cached_result, "from_cache": True, "error": None}

        async with self.semaphore:
            for attempt in range(self.api_config.retry_attempts):
                try:
                    # mention
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",
                         "content": f"分析以下评论：{text}\n注意：mention要简短概括（5-10字），不要复制原文。"}
                    ]

                    response = await self.client.chat.completions.create(
                        model=self.api_config.model,
                        messages=messages,
                        tools=[self.tool],
                        tool_choice={"type": "function", "function": {"name": "analyze_urban_perception"}}
                    )

                    message = response.choices[0].message

                    if message.tool_calls:
                        tool_call = message.tool_calls[0]
                        analysis_args = json.loads(tool_call.function.arguments)

                        # result
                        final_result = {
                            "sentiment_score": analysis_args.get("sentiment_score"),
                            "sentiment_category": analysis_args.get("sentiment_category"),
                            "urban_aspects": analysis_args.get("urban_aspects", [])
                        }

                        self._save_to_cache(cache_key, final_result)

                        aspect_count = len(final_result["urban_aspects"])
                        logger.info(f"行 {row_index}: {final_result['sentiment_category'][:3]} {aspect_count}要素")

                        return {
                            "row_index": row_index,
                            "success": True,
                            "result": final_result,
                            "from_cache": False,
                            "error": None
                        }
                    else:
                        error_note = "工具未调用"
                        logger.warning(f"行 {row_index}: {error_note}")
                        return {
                            "row_index": row_index,
                            "success": False,
                            "result": None,
                            "from_cache": False,
                            "error": error_note
                        }

                except Exception as e:
                    logger.warning(f" {row_index}: {attempt + 1}error: {e}")
                    if attempt < self.api_config.retry_attempts - 1:
                        await asyncio.sleep(self.api_config.retry_delay * (attempt + 1))
                    else:
                        return {
                            "row_index": row_index,
                            "success": False,
                            "result": None,
                            "from_cache": False,
                            "error": str(e)
                        }

            return {
                "row_index": row_index,
                "success": False,
                "result": None,
                "from_cache": False,
                "error": "最大重试次数"
            }

    async def process_batch(self) -> pd.DataFrame:
        """batch"""
        df = self.load_data()
        processed_indices = self.load_processed_indices()
        rows_to_process = [(idx, str(row[self.task_config.text_column]))
                           for idx, row in df.iterrows()
                           if idx not in processed_indices]

        if not rows_to_process:
            logger.info("finish")
            return self.compile_results(df)

        logger.info(f": {len(rows_to_process)}/{len(df)} ")
        total_processed, api_calls, cache_hits = 0, 0, 0

        for batch_start in range(0, len(rows_to_process), self.task_config.batch_size):
            batch_end = min(batch_start + self.task_config.batch_size, len(rows_to_process))
            batch = rows_to_process[batch_start:batch_end]
            logger.info(f" {batch_start // self.task_config.batch_size + 1}: {batch_start + 1}-{batch_end}")

            tasks = [self.analyze_single(text, idx) for idx, text in batch]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if result.get('from_cache'):
                    cache_hits += 1
                elif result.get('success'):
                    api_calls += 1
                    self.save_checkpoint(result['row_index'], result['result'])
                total_processed += 1

            logger.info(f": {total_processed}/{len(rows_to_process)} 行")

        if self.api_config.enable_cache:
            self.save_cache()
            total_calls = api_calls + cache_hits
            if total_calls > 0:
                logger.info(
                    f": API {api_calls} , huancun {cache_hits} ,  {cache_hits / total_calls * 100:.1f}%")

        return self.compile_results(df)

    def compile_results(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """"""
        results_dict = {}
        checkpoint_path = Path(self.task_config.resume_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            row_idx = data.get('row_index')
                            if row_idx is not None:
                                results_dict[row_idx] = data.get('result', {})
            except Exception as e:
                logger.error(f"error: {e}")

        result_columns = {
            'sentiment_score': [], 'sentiment_category': [], 'urban_aspects': [],
            'analysis_success': [], 'error_message': []
        }

        for idx in range(len(original_df)):
            if idx in results_dict:
                result = results_dict[idx]
                result_columns['sentiment_score'].append(result.get('sentiment_score'))
                result_columns['sentiment_category'].append(result.get('sentiment_category'))
                result_columns['urban_aspects'].append(json.dumps(result.get('urban_aspects', []), ensure_ascii=False))
                result_columns['analysis_success'].append(True)
                result_columns['error_message'].append(None)
            else:
                for col in result_columns:
                    if col == 'analysis_success':
                        result_columns[col].append(False)
                    elif col == 'error_message':
                        result_columns[col].append("no")
                    else:
                        result_columns[col].append(None)

        result_df = original_df.copy()
        for col_name, col_data in result_columns.items():
            result_df[f'analysis_{col_name}'] = col_data
        return result_df

    def save_results(self, df: pd.DataFrame):
        """save result"""
        output_path = Path(self.task_config.output_file)
        try:
            if output_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"save: {output_path}")

            if 'analysis_analysis_success' in df.columns:
                success_count = df['analysis_analysis_success'].sum()
                total_count = len(df)
                logger.info(f"s: {total_count} , success {success_count} ")

            if 'analysis_sentiment_category' in df.columns:
                dist = df['analysis_sentiment_category'].value_counts()
                logger.info("sentiment:")
                for k, v in dist.items():
                    logger.info(f"  {k}: {v}")

        except Exception as e:
            logger.error(f"error: {e}")
            raise


#
async def main():
    """main"""
    api_config = DeepSeekConfig(
        api_key="The ApI key number is private and cannot be made public",  # your API key
        model="deepseek-chat",
        max_concurrent=5,
        enable_cache=True,
        cache_file="deepseek_optimized_cache.json",
        cache_ttl=7 * 24 * 3600
    )

    #
    task_config = SentimentConfig(
        input_file="data/shenzhen/result/shenzhenqd_cleaned2.xlsx",
        output_file="data/shenzhen/result/checkin_result2.xlsx",
        text_column="text",
        max_rows=None,  #
        batch_size=20,
        resume_file="optimized_progress.jsonl",
        use_strict_mode=True,
        binary_classification=True
    )

    analyzer = OptimizedUrbanAnalyzer(api_config, task_config)
    result_df = await analyzer.process_batch()
    analyzer.save_results(result_df)


if __name__ == "__main__":
    asyncio.run(main())