"""Telegram-бот, HTTP API и веб-сервер Mini App omnia."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
    Update,
    WebAppInfo,
)
from dotenv import load_dotenv

from database import create_database


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
APP_URL = (
    os.getenv("APP_URL", "").strip()
    or os.getenv("RENDER_EXTERNAL_URL", "").strip()
).rstrip("/")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "omnia.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()
WEBHOOK_MODE = os.getenv("WEBHOOK_MODE", "false").lower() == "true"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.gen-api.ru/api/v1/networks/deepseek-v4").strip().rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-v4-flash").strip()
AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "55"))
MANUAL_PREMIUM_USER_IDS = {
    *{
        int(value)
        for value in os.getenv("MANUAL_PREMIUM_USER_IDS", "").replace(";", ",").split(",")
        if value.strip().isdigit()
    },
}

PRICE_STARS = 200
PAYLOAD_PREFIX = "omnia-premium"

database = create_database(DATABASE_URL or DATABASE_PATH)
router = Router()


def ai_endpoint_url() -> str:
    """Return the configured AI endpoint.

    GenAPI's native DeepSeek V4 endpoint is more reliable for this model than
    the OpenAI-compatible proxy. If an old proxy URL is still present in Render,
    route it to the native endpoint automatically.
    """
    explicit_endpoint = os.getenv("AI_ENDPOINT", "").strip().rstrip("/")
    if explicit_endpoint:
        return explicit_endpoint
    if AI_BASE_URL.startswith("https://proxy.gen-api.ru"):
        return "https://api.gen-api.ru/api/v1/networks/deepseek-v4"
    return AI_BASE_URL


def ai_model_name() -> str:
    """Normalize GenAPI page slug to a real DeepSeek V4 model version."""
    if AI_MODEL == "deepseek-v4":
        return "deepseek-v4-flash"
    return AI_MODEL


def extract_ai_message_content(data: Any) -> str:
    """Support OpenAI-compatible and GenAPI native response shapes."""
    if not isinstance(data, dict):
        raise RuntimeError("ИИ вернула ответ неизвестного формата")

    error = data.get("error") or data.get("message_error")
    if error:
        raise RuntimeError(f"GenAPI вернул ошибку: {error}")

    status = str(data.get("status") or "").lower()
    if status in {"failed", "error", "canceled", "cancelled"}:
        raise RuntimeError(f"GenAPI вернул статус {status}")

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if content:
            return str(content).strip()

    output = data.get("output")
    if isinstance(output, str) and output.strip():
        return output.strip()
    if isinstance(output, list) and output:
        output_text = extract_text_from_nested_result(output)
        if output_text:
            return output_text
    if isinstance(output, dict):
        output_text = extract_text_from_nested_result(output)
        if output_text:
            return output_text

    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    if isinstance(response, dict):
        response_text = extract_text_from_nested_result(response)
        if response_text:
            return response_text

    result = data.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    if isinstance(result, (dict, list)):
        result_text = extract_text_from_nested_result(result)
        if result_text:
            return result_text

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(text, list) and text:
        return "\n".join(str(item) for item in text if item).strip()

    # В синхронных ответах GenAPI итог языковой модели часто находится здесь,
    # а поле result при этом остаётся пустым массивом.
    for key in ("full_response", "data", "payload"):
        nested_text = extract_text_from_nested_result(data.get(key))
        if nested_text:
            return nested_text

    logging.error("AI response contains no text. Shape: %s", describe_response_shape(data))

    raise RuntimeError("ИИ вернула ответ без текста сценария")


async def poll_genapi_result(
    request_id: Any,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Дождаться результата GenAPI, если первый ответ содержит только request_id."""
    safe_request_id = quote(str(request_id), safe="")
    result_url = f"https://api.gen-api.ru/api/v1/request/get/{safe_request_id}"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Accept": "application/json",
    }
    deadline = time.monotonic() + (timeout_seconds or AI_TIMEOUT_SECONDS)
    last_data: dict[str, Any] = {"request_id": request_id, "status": "processing"}

    async with ClientSession(timeout=ClientTimeout(total=30)) as session:
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            async with session.get(result_url, headers=headers) as response:
                response_text = await response.text()
                if response.status >= 400:
                    logging.error(
                        "GenAPI result API returned %s: %s",
                        response.status,
                        response_text[:500],
                    )
                    raise RuntimeError(f"GenAPI не смогла выдать результат: ошибка {response.status}")
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as error:
                    logging.error("GenAPI result is not JSON: %s", response_text[:500])
                    raise RuntimeError("GenAPI вернула неизвестный формат результата") from error

            if not isinstance(data, dict):
                continue
            last_data = data
            status = str(data.get("status") or "").lower()
            if status in {"failed", "error", "canceled", "cancelled"}:
                detail = data.get("error") or data.get("message") or status
                raise RuntimeError(f"GenAPI не завершила генерацию: {detail}")
            if status == "success":
                return data

            # Некоторые ответы уже содержат текст до обновления поля status.
            try:
                extract_ai_message_content(data)
                return data
            except RuntimeError:
                pass

    logging.error("GenAPI polling timed out. Shape: %s", describe_response_shape(last_data))
    raise RuntimeError("GenAPI слишком долго генерирует сценарий. Попробуйте ещё раз")


def extract_text_from_nested_result(value: Any) -> str:
    """Find generated text inside GenAPI's nested task response."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [extract_text_from_nested_result(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if not isinstance(value, dict):
        return ""

    for key in (
        "content",
        "text",
        "message",
        "answer",
        "response",
        "result",
        "output",
        "full_response",
        "data",
        "payload",
        "completion",
        "generated_text",
    ):
        nested = value.get(key)
        text = extract_text_from_nested_result(nested)
        if text:
            return text

    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        return extract_text_from_nested_result(choices[0])

    return ""


def describe_response_shape(value: Any, depth: int = 0) -> Any:
    """Описать структуру ответа для Render Logs, не записывая текст пользователя."""
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key): describe_response_shape(nested, depth + 1)
            for key, nested in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [describe_response_shape(item, depth + 1) for item in value[:3]]
    if isinstance(value, str):
        return f"str({len(value)})"
    return type(value).__name__


def validate_init_data(init_data: str) -> dict[str, Any] | None:
    """Проверить подпись Telegram Web App и вернуть пользователя."""
    if not init_data or not BOT_TOKEN:
        return None

    values = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = values.pop("hash", "")
    if not received_hash:
        return None

    data_check_string = "\n".join(
        f"{key}={value}" for key, value in sorted(values.items())
    )
    secret_key = hmac.new(
        b"WebAppData", BOT_TOKEN.encode("utf-8"), hashlib.sha256
    ).digest()
    calculated_hash = hmac.new(
        secret_key, data_check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(calculated_hash, received_hash):
        return None

    # Не принимаем авторизацию, которая была выдана более суток назад.
    try:
        auth_date = int(values.get("auth_date", "0"))
        if abs(int(time.time()) - auth_date) > 86_400:
            return None
        user = json.loads(values["user"])
        if not isinstance(user.get("id"), int):
            return None
        return user
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


@web.middleware
async def authentication_middleware(
    request: web.Request, handler: Any
) -> web.StreamResponse:
    """Защитить все методы API проверенной Telegram-авторизацией."""
    if not request.path.startswith("/api/"):
        return await handler(request)

    telegram_user = validate_init_data(
        request.headers.get("X-Telegram-Init-Data", "")
    )

    # Режим нужен только для локальной проверки страницы вне Telegram.
    if telegram_user is None and DEVELOPMENT_MODE:
        development_user_id = request.headers.get("X-Development-User-Id", "100001")
        if development_user_id.isdigit():
            telegram_user = {
                "id": int(development_user_id),
                "username": "local_preview",
                "first_name": "Миша",
            }

    if telegram_user is None:
        return web.json_response(
            {"error": "Откройте приложение из Telegram"}, status=401
        )

    request["telegram_user"] = telegram_user
    return await handler(request)


def current_user(request: web.Request) -> dict[str, Any]:
    return request["telegram_user"]


def ensure_premium_access(user_id: int, username: str | None) -> dict[str, Any]:
    """Create/update a user and apply manual Premium grants configured by owner."""
    profile = database.ensure_user(user_id, username)
    if user_id in MANUAL_PREMIUM_USER_IDS and not profile["is_premium"]:
        database.activate_premium(
            user_id=user_id,
            payment_charge_id=f"manual-premium-{user_id}",
            invoice_payload="manual-premium-grant",
            amount=0,
            currency="MANUAL",
        )
        profile = database.get_user(user_id) or profile
    return profile


def format_video_time(total_seconds: int) -> str:
    """Форматировать время ролика без случайного округления."""
    minutes, seconds = divmod(max(0, total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def video_scene_count(duration: str, duration_minutes: int | None) -> int:
    if duration == "short":
        return 3
    minutes = max(1, min(60, duration_minutes or 10))
    return max(3, min(12, (minutes + 4) // 5 + 2))


def exact_video_timings(duration: str, scene_count: int, duration_minutes: int | None) -> list[str]:
    if duration == "short":
        return ["00:00–00:04", "00:04–00:43", "00:43–01:00"]
    total_seconds = max(1, min(60, duration_minutes or 10)) * 60
    return [
        f"{format_video_time(total_seconds * index // scene_count)}–"
        f"{format_video_time(total_seconds * (index + 1) // scene_count)}"
        for index in range(scene_count)
    ]


def text_word_count(text: str) -> int:
    """Посчитать произносимые слова в русском или английском тексте."""
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-’'][A-Za-zА-Яа-яЁё0-9]+)*", text))


def realistic_video_timings(
    scenes: list[dict[str, Any]],
    duration: str,
    duration_minutes: int | None,
    language: str,
) -> list[str]:
    """Рассчитать тайминг по объёму речи и реальным паузам/B-roll.

    Небольшое отклонение от целевой длины мягко компенсируется паузами. Если
    модель написала слишком мало текста, интерфейс честно покажет более
    короткий хронометраж и не растянет одну минуту речи на несколько минут.
    """
    if duration == "short":
        return exact_video_timings(duration, len(scenes), duration_minutes)

    words_per_minute = 145 if language == "en" else 125
    durations: list[int] = []
    for scene in scenes:
        speech_seconds = max(4, round(text_word_count(str(scene.get("speaker", ""))) * 60 / words_per_minute))
        try:
            visual_seconds = int(scene.get("visual_seconds", 0))
        except (TypeError, ValueError):
            visual_seconds = 0
        # Паузы и перебивки не могут маскировать слишком короткий текст.
        visual_seconds = max(2, min(visual_seconds, max(12, speech_seconds // 3)))
        durations.append(speech_seconds + visual_seconds)

    target_seconds = max(1, min(60, duration_minutes or 10)) * 60
    natural_seconds = sum(durations)
    if natural_seconds and target_seconds * 0.85 <= natural_seconds <= target_seconds * 1.08:
        factor = target_seconds / natural_seconds
        durations = [max(1, round(value * factor)) for value in durations]
        durations[-1] += target_seconds - sum(durations)

    timings: list[str] = []
    cursor = 0
    for seconds in durations:
        end = cursor + max(1, seconds)
        timings.append(f"{format_video_time(cursor)}–{format_video_time(end)}")
        cursor = end
    return timings


def normalize_ai_scenes(
    raw_scenes: list[dict[str, Any]],
    duration: str,
    duration_minutes: int | None,
    language: str,
) -> list[dict[str, str]]:
    timings = realistic_video_timings(raw_scenes, duration, duration_minutes, language)
    fallback_titles = ["Hook", "Development", "Finale"] if language == "en" else ["Хук", "Развитие", "Финал"]
    scenes: list[dict[str, str]] = []
    for index, scene in enumerate(raw_scenes):
        fallback = fallback_titles[index] if len(raw_scenes) == 3 else (f"Scene {index + 1}" if language == "en" else f"Смысловой блок {index + 1}")
        scenes.append(
            {
                "title": str(scene.get("title") or fallback).strip()[:80],
                "timing": timings[index],
                "frame": str(scene.get("frame") or "").strip(),
                "speaker": str(scene.get("speaker") or "").strip(),
                "light": str(scene.get("light") or "").strip(),
                "sound": str(scene.get("sound") or "").strip(),
            }
        )
    return scenes


def extract_json_from_ai_answer(text: str) -> Any:
    """Достать JSON даже если модель обернула его в markdown-блок."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    # Некоторые модели оборачивают массив в объект, даже если попросить
    # вернуть только JSON-массив.
    if isinstance(parsed, dict):
        for key in ("scenes", "script", "scenario", "blocks", "items"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                return candidate
    return parsed


async def generate_ai_script(
    topic: str,
    duration: str,
    tone: str,
    duration_minutes: int | None = None,
    language: str = "ru",
    options: dict[str, str] | None = None,
    existing_content: list[dict[str, str]] | None = None,
    revision_instruction: str = "",
) -> list[dict[str, str]]:
    """Сгенерировать сценарий через GenAPI/OpenAI-compatible endpoint."""
    if not AI_API_KEY:
        return make_script(topic, duration, tone, duration_minutes, language)

    language = "en" if language == "en" else "ru"
    exact_minutes = max(1, min(60, duration_minutes or 10))
    scene_count = video_scene_count(duration, exact_minutes)
    target_words = (135 if language == "en" else 115) if duration == "short" else (120 if language == "en" else 95) * exact_minutes
    target_visual_seconds = 6 if duration == "short" else 12 * exact_minutes
    duration_label = "Shorts/Reels exactly 60 seconds" if language == "en" and duration == "short" else f"YouTube video exactly {exact_minutes} minutes" if language == "en" else "Shorts/Reels ровно на 60 секунд" if duration == "short" else f"YouTube-видео ровно на {exact_minutes} минут"
    tone_labels = {
        "ru": {"hype": "динамичный хайп, быстро, энергично, без воды", "emotional": "эмоциональный, искренний, мотивирующий", "educational": "обучающий, практичный, пошаговый"},
        "en": {"hype": "dynamic and energetic, fast-paced, no filler", "emotional": "emotional, sincere and motivating", "educational": "educational, practical and step-by-step"},
    }
    tone_label = tone_labels[language][tone]
    options = options or {}
    option_labels = ({"platform": "Platform", "audience": "Target audience", "goal": "Video goal", "format": "Shooting format", "cta": "Call to action"} if language == "en" else {"platform": "Площадка", "audience": "Целевая аудитория", "goal": "Цель видео", "format": "Формат съёмки", "cta": "Призыв к действию"})
    brief_lines = [
        f"{option_labels[key]}: {value}"
        for key, value in options.items()
        if key in option_labels and value
    ]
    brief = "\n".join(brief_lines) or ("No additional preferences" if language == "en" else "Дополнительных пожеланий нет")
    revision_context = ""
    if existing_content and revision_instruction:
        revision_context = (f"\n\nImprove this existing script.\nCurrent script: {json.dumps(existing_content, ensure_ascii=False)}\nRevision task: {revision_instruction}\nKeep the successful details and the structure of unaffected scenes." if language == "en" else f"\n\nЭто улучшение уже существующего сценария.\nТекущий сценарий: {json.dumps(existing_content, ensure_ascii=False)}\nЗадача улучшения: {revision_instruction}\nСохрани удачные детали и структуру остальных блоков.")

    if language == "en":
        system_prompt = "You are an expert English-language video scriptwriter and editing director. Be concrete, useful and natural. Return only valid JSON with no explanation."
        user_prompt = f"""
Create a script about: {topic}
Format: {duration_label}
Tone: {tone_label}
Brief:
{brief}
{revision_context}

Important:
- Write the entire result in English, including titles and production directions.
- Avoid generic filler. Give real actions, examples, mistakes and visual ideas.
- The speaker text must sound natural when spoken aloud.
- Return exactly a JSON array with {scene_count} scenes.
- For this duration write about {target_words} spoken words total (within 8%). This is mandatory; do not replace speech with a short summary.
- Distribute approximately {target_visual_seconds} seconds of pauses, demonstrations and B-roll across the scenes.
- Include integer `visual_seconds` in each scene for time with no spoken words.
- Timings must cover the narrative continuously, but the server will verify them from the actual word count.

Scene schema:
{{
  "title": "scene title",
  "timing": "proposed timing",
  "frame": "what happens on screen",
  "speaker": "the complete verbatim text to be spoken, not an outline",
  "light": "lighting atmosphere",
  "sound": "music and sound design",
  "visual_seconds": 10
}}
""".strip()
    else:
        system_prompt = "Ты сильный русскоязычный сценарист видео и режиссёр монтажа. Пиши конкретно, применимо и естественно. Не объясняй формат, верни только валидный JSON."
        user_prompt = f"""
Сделай сценарий для темы: {topic}
Формат: {duration_label}
Тональность: {tone_label}
Бриф:
{brief}
{revision_context}

Важно:
- Никакой воды вроде "главная деталь темы" или "три опоры".
- Дай реальные действия, упражнения, ошибки, примеры и визуальные кадры.
- Текст спикера должен звучать как живой человек из Reels/TikTok.
- Верни ровно JSON-массив из {scene_count} смысловых блоков.
- Для этой длительности напиши около {target_words} произносимых слов суммарно (допуск 8%). Это обязательное условие: нужен полный текст, а не краткий конспект.
- Распредели примерно {target_visual_seconds} секунд пауз, демонстраций и B-roll между блоками.
- В каждом блоке укажи целое поле `visual_seconds` — время без речи.
- Тайминг должен быть непрерывным, но сервер перепроверит его по фактическому количеству слов.

Схема каждого блока:
{{
  "title": "название блока",
  "timing": "тайминг",
  "frame": "что происходит в кадре",
  "speaker": "полный дословный текст спикера, а не тезисы",
  "light": "световая атмосфера",
  "sound": "звуки и эффекты",
  "visual_seconds": 10
}}
""".strip()

    payload = {
        "is_sync": True,
        "model": ai_model_name(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.85,
        "max_tokens": (
            1800
            if duration == "short"
            else min(8000, 2200 + exact_minutes * 95)
        ),
    }

    generation_timeout = (
        AI_TIMEOUT_SECONDS
        if duration == "short"
        else max(AI_TIMEOUT_SECONDS, 70 + exact_minutes * 2)
    )
    timeout = ClientTimeout(total=generation_timeout)
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(ai_endpoint_url(), headers=headers, json=payload) as response:
                response_text = await response.text()
                if response.status >= 400:
                    logging.error("AI API returned %s: %s", response.status, response_text[:500])
                    detail = response_text[:240]
                    try:
                        error_payload = json.loads(response_text)
                        raw_detail = (
                            error_payload.get("errors")
                            or error_payload.get("detail")
                            or error_payload.get("error")
                            or error_payload.get("message")
                            or error_payload
                        )
                        if isinstance(raw_detail, (dict, list)):
                            detail = json.dumps(raw_detail, ensure_ascii=False)[:240]
                        else:
                            detail = str(raw_detail)[:240]
                    except json.JSONDecodeError:
                        pass
                    raise RuntimeError(f"GenAPI вернул ошибку {response.status}: {detail}")
                data = json.loads(response_text)
    except RuntimeError:
        raise
    except (ClientError, TimeoutError, json.JSONDecodeError) as error:
        logging.error("AI generation failed: %s", error)
        raise RuntimeError("ИИ сейчас не ответила. Проверьте AI_API_KEY, AI_BASE_URL и AI_MODEL в Render.") from error

    try:
        answer = extract_ai_message_content(data)
    except RuntimeError as error:
        request_id = data.get("request_id") or data.get("id") if isinstance(data, dict) else None
        if not request_id:
            raise error
        logging.info("GenAPI queued request %s; polling for result", request_id)
        data = await poll_genapi_result(request_id, generation_timeout)
        answer = extract_ai_message_content(data)

    try:
        parsed = extract_json_from_ai_answer(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        logging.error("AI response is not valid JSON: %s", error)
        raise RuntimeError("ИИ вернула ответ не в JSON-формате") from error

    if not isinstance(parsed, list) or len(parsed) < 3:
        logging.error("AI response has invalid scene list")
        raise RuntimeError("ИИ вернула неправильную структуру сценария")

    available_count = min(scene_count, len(parsed))
    scenes = normalize_ai_scenes(parsed[:available_count], duration, exact_minutes, language)
    if any(not scene["speaker"] or not scene["frame"] for scene in scenes):
        logging.error("AI response has empty required scene fields")
        raise RuntimeError("ИИ вернула неполный сценарий")
    return scenes


def make_script(
    topic: str,
    duration: str,
    tone: str,
    duration_minutes: int | None = None,
    language: str = "ru",
) -> list[dict[str, str]]:
    """Сформировать готовый структурированный сценарий без внешнего API."""
    is_short = duration == "short"
    exact_minutes = max(1, min(60, duration_minutes or 10))
    duration_label = "60 секунд" if is_short else f"{exact_minutes} минут"
    opening_timing, body_timing, finale_timing = exact_video_timings(
        duration,
        3,
        exact_minutes,
    )

    if language == "en":
        tone_copy = {"hype": "speak quickly and confidently, building energy", "emotional": "speak sincerely with personal emotion and short pauses", "educational": "explain clearly, precisely and step by step"}[tone]
        scenes = [
            {"title": "Instant hook", "timing": opening_timing, "frame": f"Macro shot of a concrete detail connected to “{topic}”, then a sharp cut to the presenter.", "speaker": f"If you think you already know everything about {topic}, give me {duration_label}. The most important part is usually overlooked.", "light": "Blue neon rim light, face in partial shadow, soft warm fill from the right.", "sound": "Short cinematic hit followed by a pulsing bass and a clean whoosh."},
            {"title": "Value and development", "timing": body_timing, "frame": "Alternate medium shots, practical details and concise on-screen points.", "speaker": f"Let us break down {topic} into concrete actions. Set one measurable goal, practise it consistently, record the result and review what actually changed. We {tone_copy}. Show the viewer a real before-and-after result instead of a vague promise.", "light": "Soft key light at 45 degrees with a blue-violet background gradient.", "sound": "Rhythmic electronic underscore with subtle clicks between key points."},
            {"title": "Finale and action", "timing": finale_timing, "frame": "The camera slowly moves closer while one clear call to action appears.", "speaker": f"The main idea is simple: {topic} becomes achievable when you turn it into the next concrete action. Save this script and take that action today.", "light": "The neon becomes softer and warmer while the background fades to deep black.", "sound": "The music resolves into a bright chord and a clean final impact."},
        ]
        return normalize_ai_scenes(scenes, duration, exact_minutes, language) if not is_short else scenes

    tone_copy = {
        "hype": "говорим быстро, уверенно и с нарастающей энергией",
        "emotional": "говорим искренне, через личное переживание и короткие паузы",
        "educational": "объясняем просто, точно и по шагам",
    }[tone]

    scenes = [
        {
            "title": "Мгновенный хук",
            "timing": opening_timing,
            "frame": f"Макро-план главной детали темы «{topic}». Резкий переход на лицо автора и крупный заголовок.",
            "speaker": f"Если вы думаете, что уже всё знаете про «{topic}», дайте мне {duration_label}. Самое важное обычно упускают.",
            "light": "Контровой неоновый синий свет, лицо в полутени, мягкий тёплый заполняющий источник справа.",
            "sound": "Короткий cinematic hit, затем пульсирующий бас; акцентный whoosh на появлении заголовка.",
        },
        {
            "title": "Раскрытие и ценность",
            "timing": body_timing,
            "frame": "Чередование среднего плана, деталей и лаконичных экранных тезисов. Каждый новый смысл поддержан сменой ракурса.",
            "speaker": f"Разберём тему «{topic}» на три опоры. Первая — понятная цель. Вторая — один измеримый шаг. Третья — обратная связь. Здесь {tone_copy}. Покажите зрителю не обещание, а конкретное изменение до и после.",
            "light": "Мягкий ключевой свет под 45 градусов, сине-фиолетовый градиент на фоне, тонкий контур на плечах.",
            "sound": "Ритмичный electronic underscore без вокала, тихие клики на смене тезисов, лёгкий riser перед выводом.",
        },
        {
            "title": "Финал и действие",
            "timing": finale_timing,
            "frame": "Камера медленно приближается. На финальной фразе появляется один крупный призыв к действию.",
            "speaker": f"Главная мысль проста: «{topic}» начинает работать, когда вы превращаете идею в следующий конкретный шаг. Сохраните сценарий и попробуйте сегодня.",
            "light": "Неон становится мягче и теплее, взгляд подсвечен, фон уходит в глубокий чёрный для фокуса на авторе.",
            "sound": "Музыка раскрывается светлым аккордом, короткий reverse-переход и чистый финальный удар под призыв.",
        },
    ]
    return normalize_ai_scenes(scenes, duration, exact_minutes, language) if not is_short else scenes


async def api_profile(request: web.Request) -> web.Response:
    user = current_user(request)
    profile = await asyncio.to_thread(
        ensure_premium_access,
        user["id"],
        user.get("username") or user.get("first_name"),
    )
    return web.json_response({"profile": profile})


async def api_scenarios(request: web.Request) -> web.Response:
    user = current_user(request)
    scenarios = await asyncio.to_thread(database.list_scenarios, user["id"])
    return web.json_response({"scenarios": scenarios})


def scenario_id_from_request(request: web.Request) -> int:
    try:
        return int(request.match_info["scenario_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise web.HTTPBadRequest(text="Некорректный идентификатор сценария") from error


async def api_update_scenario(request: web.Request) -> web.Response:
    user = current_user(request)
    scenario_id = scenario_id_from_request(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Некорректный запрос"}, status=400)

    topic = body.get("topic")
    favorite = body.get("is_favorite")
    if topic is not None:
        topic = str(topic).strip()
        if not 3 <= len(topic) <= 180:
            return web.json_response({"error": "Название должно содержать от 3 до 180 символов"}, status=400)
    if favorite is not None and not isinstance(favorite, bool):
        return web.json_response({"error": "Некорректное значение избранного"}, status=400)

    scenario = await asyncio.to_thread(
        database.update_scenario,
        user["id"],
        scenario_id,
        topic=topic,
        is_favorite=favorite,
    )
    if scenario is None:
        return web.json_response({"error": "Сценарий не найден"}, status=404)
    return web.json_response({"scenario": scenario})


async def api_delete_scenario(request: web.Request) -> web.Response:
    user = current_user(request)
    scenario_id = scenario_id_from_request(request)
    deleted = await asyncio.to_thread(database.delete_scenario, user["id"], scenario_id)
    if not deleted:
        return web.json_response({"error": "Сценарий не найден"}, status=404)
    return web.json_response({"deleted": True})


async def api_refine_scenario(request: web.Request) -> web.Response:
    user = current_user(request)
    scenario_id = scenario_id_from_request(request)
    profile = await asyncio.to_thread(
        ensure_premium_access,
        user["id"],
        user.get("username") or user.get("first_name"),
    )
    if not profile["is_premium"]:
        return web.json_response(
            {"error": "Улучшение отдельных частей доступно в Premium", "paywall": True},
            status=402,
        )
    scenario = await asyncio.to_thread(database.get_scenario, user["id"], scenario_id)
    if scenario is None:
        return web.json_response({"error": "Сценарий не найден"}, status=404)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Некорректный запрос"}, status=400)
    action = str(body.get("action", ""))
    requested_tone = str(body.get("tone", scenario["tone"]))
    if requested_tone not in {"hype", "emotional", "educational"}:
        requested_tone = scenario["tone"]
    instructions = {
        "stronger_hook": "Перепиши только первый блок: сделай хук конкретнее, неожиданнее и сильнее. Остальные блоки почти не меняй.",
        "shorter": "Сократи текст спикера во всех блоках примерно на 30 процентов, сохранив смысл, факты и естественный ритм.",
        "new_finale": "Перепиши только финальный блок: усили вывод и сделай призыв к действию конкретным и ненавязчивым.",
        "change_tone": "Перепиши текст спикера под выбранную тональность, сохрани факты, тайминг и логику кадров.",
    }
    if action not in instructions:
        return web.json_response({"error": "Неизвестное действие улучшения"}, status=400)
    scenario_language = "en" if scenario.get("language") == "en" else "ru"
    if scenario_language == "en":
        instructions = {
            "stronger_hook": "Rewrite only the first scene with a more specific, surprising and powerful hook. Keep the other scenes almost unchanged.",
            "shorter": "Shorten the speaker text in every scene by about 30 percent while preserving meaning, facts and natural rhythm.",
            "new_finale": "Rewrite only the final scene with a stronger conclusion and a specific, natural call to action.",
            "change_tone": "Rewrite the speaker text in the selected tone while preserving facts, realistic timing and visual logic.",
        }

    try:
        content = await generate_ai_script(
            scenario["topic"],
            scenario["duration"],
            requested_tone,
            duration_minutes=scenario.get("duration_minutes"),
            language=scenario_language,
            existing_content=scenario["content"],
            revision_instruction=instructions[action],
        )
    except RuntimeError as error:
        return web.json_response({"error": str(error)}, status=502)
    updated = await asyncio.to_thread(
        database.update_scenario,
        user["id"],
        scenario_id,
        tone=requested_tone if action == "change_tone" else None,
        content=content,
    )
    return web.json_response({"scenario": updated, "profile": profile})


async def api_generate(request: web.Request) -> web.Response:
    user = current_user(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Некорректный запрос"}, status=400)

    topic = str(body.get("topic", "")).strip()
    duration = str(body.get("duration", ""))
    tone = str(body.get("tone", ""))
    language = "en" if str(body.get("language", "ru")).lower() == "en" else "ru"
    try:
        duration_minutes = int(body.get("duration_minutes", 10))
    except (TypeError, ValueError):
        duration_minutes = 10
    raw_options = body.get("options") if isinstance(body.get("options"), dict) else {}
    options = {
        key: str(raw_options.get(key, "")).strip()[:180]
        for key in ("platform", "audience", "goal", "format", "cta")
        if str(raw_options.get(key, "")).strip()
    }
    if not 3 <= len(topic) <= 180:
        return web.json_response(
            {"error": "Введите тему длиной от 3 до 180 символов"}, status=400
        )
    if duration not in {"short", "long"} or tone not in {
        "hype",
        "emotional",
        "educational",
    }:
        return web.json_response({"error": "Проверьте параметры сценария"}, status=400)
    if duration == "long" and not 1 <= duration_minutes <= 60:
        return web.json_response(
            {"error": "Длительность YouTube-видео должна быть от 1 до 60 минут"},
            status=400,
        )
    if duration == "short":
        duration_minutes = None

    await asyncio.to_thread(
        ensure_premium_access,
        user["id"],
        user.get("username") or user.get("first_name"),
    )
    profile = await asyncio.to_thread(database.consume_generation, user["id"])
    if profile is None:
        return web.json_response(
            {"error": "Бесплатные генерации закончились", "paywall": True},
            status=402,
        )

    try:
        content = await generate_ai_script(
            topic,
            duration,
            tone,
            duration_minutes=duration_minutes,
            language=language,
            options=options,
        )
    except RuntimeError as error:
        return web.json_response({"error": str(error)}, status=502)
    scenario_id = await asyncio.to_thread(
        database.save_scenario,
        user["id"],
        topic,
        duration,
        tone,
        content,
        duration_minutes,
        language,
    )
    return web.json_response(
        {
            "scenario": {
                "id": scenario_id,
                "topic": topic,
                "duration": duration,
                "duration_minutes": duration_minutes,
                "language": language,
                "tone": tone,
                "content": content,
                "is_favorite": False,
            },
            "profile": profile,
        }
    )


async def api_invoice(request: web.Request) -> web.Response:
    user = current_user(request)
    profile = await asyncio.to_thread(
        ensure_premium_access,
        user["id"],
        user.get("username") or user.get("first_name"),
    )
    if profile["is_premium"]:
        return web.json_response({"already_premium": True})

    payload = f"{PAYLOAD_PREFIX}:{user['id']}:{secrets.token_urlsafe(10)}"
    bot: Bot = request.app["bot"]
    invoice_link = await bot.create_invoice_link(
        title="Подписка VibeScript AI",
        description="Безлимитная генерация и сохранение сценариев в omnia",
        payload=payload,
        currency="XTR",
        prices=[LabeledPrice(label="Подписка VibeScript AI", amount=PRICE_STARS)],
    )
    return web.json_response({"invoice_url": invoice_link})


async def index_page(_: web.Request) -> web.FileResponse:
    return web.FileResponse(BASE_DIR / "index.html")


async def health(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "ai_enabled": bool(AI_API_KEY),
            "ai_model": AI_MODEL,
            "ai_effective_model": ai_model_name(),
            "ai_base_url": AI_BASE_URL,
            "ai_endpoint": ai_endpoint_url(),
        }
    )


async def telegram_webhook(request: web.Request) -> web.Response:
    """Принять обновление Telegram на спящем бесплатном web-сервисе."""
    expected_secret = request.app["webhook_secret"]
    received_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not expected_secret or not hmac.compare_digest(received_secret, expected_secret):
        return web.json_response({"error": "Недействительная подпись webhook"}, status=403)

    try:
        update = Update.model_validate(await request.json())
    except (ValueError, json.JSONDecodeError):
        return web.json_response({"error": "Некорректное обновление"}, status=400)

    dispatcher: Dispatcher = request.app["dispatcher"]
    bot: Bot = request.app["bot"]
    await dispatcher.feed_update(bot, update)
    return web.json_response({"ok": True})


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    """Показывает пользователю премиальную welcome-карточку omnia."""
    if message.from_user is None:
        return

    await asyncio.to_thread(
        ensure_premium_access,
        message.from_user.id,
        message.from_user.username,
    )
    keyboard_rows = [
        [
            InlineKeyboardButton(
                text="Открыть AI-студию",
                web_app=WebAppInfo(url=APP_URL),
            )
        ]
    ]
    keyboard_rows.append(
        [
            InlineKeyboardButton(
                text=f"Купить Premium — {PRICE_STARS} Stars",
                callback_data="buy_premium",
            )
        ]
    )

    caption = (
        "<blockquote>"
        "<b>Добро пожаловать в omnia</b>\n\n"
        "Ваша персональная AI-студия для создания видео, которые хочется досмотреть.\n\n"
        "<b>Что умеет omnia</b>\n"
        "• превращает идею в готовый сценарий\n"
        "• расписывает тайминг и действия в кадре\n"
        "• пишет естественный текст для спикера\n"
        "• подбирает свет и атмосферу\n"
        "• добавляет музыку и звуковые эффекты\n"
        "• сохраняет сценарии в вашей истории\n\n"
        "<b>Первые 3 генерации — бесплатно.</b>\n"
        f"Premium без лимитов — всего {PRICE_STARS} Stars"
        "</blockquote>"
    )
    await message.answer_photo(
        photo=FSInputFile(BASE_DIR / "assets" / "welcome-poster.png"),
        caption=caption,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard_rows),
        parse_mode="HTML",
    )


@router.callback_query(F.data == "buy_premium")
async def buy_premium_callback(callback: CallbackQuery) -> None:
    """Создаёт нативный Telegram Stars Invoice из welcome-карточки."""
    profile = await asyncio.to_thread(
        ensure_premium_access,
        callback.from_user.id,
        callback.from_user.username,
    )
    if profile["is_premium"]:
        await callback.answer("Premium уже активирован", show_alert=True)
        return
    if callback.message is None:
        await callback.answer()
        return

    await callback.answer()
    payload = f"{PAYLOAD_PREFIX}:{callback.from_user.id}:{secrets.token_urlsafe(10)}"
    await callback.message.answer_invoice(
        title="omnia Premium",
        description="Безлимитное создание и сохранение режиссёрских сценариев в omnia",
        payload=payload,
        currency="XTR",
        prices=[LabeledPrice(label="omnia Premium", amount=PRICE_STARS)],
    )


@router.pre_checkout_query()
async def pre_checkout_handler(query: PreCheckoutQuery) -> None:
    """Telegram требует ответить на запрос не позднее десяти секунд."""
    payload_parts = query.invoice_payload.split(":")
    valid_payload = (
        len(payload_parts) == 3
        and payload_parts[0] == PAYLOAD_PREFIX
        and payload_parts[1].isdigit()
        and int(payload_parts[1]) == query.from_user.id
    )
    valid_payment = query.currency == "XTR" and query.total_amount == PRICE_STARS
    if valid_payload and valid_payment:
        await query.answer(ok=True)
    else:
        await query.answer(ok=False, error_message="Не удалось проверить счёт. Создайте его заново.")


@router.message(F.successful_payment)
async def successful_payment_handler(message: Message) -> None:
    payment = message.successful_payment
    if payment is None or message.from_user is None:
        return

    payload_parts = payment.invoice_payload.split(":")
    if (
        len(payload_parts) != 3
        or payload_parts[0] != PAYLOAD_PREFIX
        or not payload_parts[1].isdigit()
        or int(payload_parts[1]) != message.from_user.id
        or payment.currency != "XTR"
        or payment.total_amount != PRICE_STARS
    ):
        logging.error("Получен платёж с некорректными параметрами: %s", payment.invoice_payload)
        return

    await asyncio.to_thread(
        database.ensure_user, message.from_user.id, message.from_user.username
    )
    await asyncio.to_thread(
        database.activate_premium,
        message.from_user.id,
        payment.telegram_payment_charge_id,
        payment.invoice_payload,
        payment.total_amount,
        payment.currency,
    )
    await message.answer(
        "Оплата прошла успешно.\n\nPremium активирован. Теперь генерации доступны без ограничений."
    )


async def run() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Укажите BOT_TOKEN в файле .env")
    if not APP_URL.startswith("https://") and not DEVELOPMENT_MODE:
        raise RuntimeError("APP_URL должен быть публичным HTTPS-адресом Mini App")
    if WEBHOOK_MODE and not WEBHOOK_SECRET:
        raise RuntimeError("В webhook-режиме укажите WEBHOOK_SECRET")

    database.initialize()
    # Если провайдер блокирует api.telegram.org, aiogram направит только
    # запросы Bot API через указанный HTTP/SOCKS-прокси.
    telegram_session = AiohttpSession(proxy=TELEGRAM_PROXY or None)
    bot = Bot(token=BOT_TOKEN, session=telegram_session)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    application = web.Application(middlewares=[authentication_middleware])
    application["bot"] = bot
    application["dispatcher"] = dispatcher
    application["webhook_secret"] = WEBHOOK_SECRET
    application.add_routes(
        [
            web.get("/", index_page),
            web.static("/assets", BASE_DIR / "assets"),
            web.static("/icons", BASE_DIR / "icons"),
            web.get("/health", health),
            web.get("/api/profile", api_profile),
            web.get("/api/scenarios", api_scenarios),
            web.patch("/api/scenarios/{scenario_id}", api_update_scenario),
            web.delete("/api/scenarios/{scenario_id}", api_delete_scenario),
            web.post("/api/scenarios/{scenario_id}/refine", api_refine_scenario),
            web.post("/api/generate", api_generate),
            web.post("/api/invoice", api_invoice),
            web.post("/telegram/webhook", telegram_webhook),
        ]
    )
    runner = web.AppRunner(application)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    logging.info("omnia запущена на %s:%s", HOST, PORT)

    try:
        if WEBHOOK_MODE:
            webhook_url = f"{APP_URL}/telegram/webhook"
            await bot.set_webhook(
                url=webhook_url,
                secret_token=WEBHOOK_SECRET,
                allowed_updates=dispatcher.resolve_used_update_types(),
            )
            logging.info("Webhook Telegram установлен: %s", webhook_url)
            await asyncio.Event().wait()
            return

        while True:
            try:
                await dispatcher.start_polling(bot, close_bot_session=False)
                break
            except TelegramNetworkError as error:
                logging.error(
                    "Нет соединения с Telegram Bot API: %s. Повтор через 5 секунд.",
                    error,
                )
                await asyncio.sleep(5)
    finally:
        await runner.cleanup()
        await bot.session.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    asyncio.run(run())
