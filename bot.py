"""Telegram-бот, HTTP API и веб-сервер Mini App omnia."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
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


async def poll_genapi_result(request_id: Any) -> dict[str, Any]:
    """Дождаться результата GenAPI, если первый ответ содержит только request_id."""
    safe_request_id = quote(str(request_id), safe="")
    result_url = f"https://api.gen-api.ru/api/v1/request/get/{safe_request_id}"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Accept": "application/json",
    }
    deadline = time.monotonic() + AI_TIMEOUT_SECONDS
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


def normalize_ai_scene(scene: dict[str, Any], index: int, duration: str) -> dict[str, str]:
    """Привести ответ модели к формату, который уже умеет рисовать фронтенд."""
    short_timings = ["00:00-00:04", "00:05-00:42", "00:43-01:00"]
    long_timings = ["00:00-00:25", "00:26-06:30", "06:31-08:00"]
    timings = short_timings if duration == "short" else long_timings
    fallback_titles = ["Хук", "Польза", "Финал"]

    return {
        "title": str(scene.get("title") or fallback_titles[index]).strip()[:80],
        "timing": str(scene.get("timing") or timings[index]).strip()[:40],
        "frame": str(scene.get("frame") or "").strip(),
        "speaker": str(scene.get("speaker") or "").strip(),
        "light": str(scene.get("light") or "").strip(),
        "sound": str(scene.get("sound") or "").strip(),
    }


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
    options: dict[str, str] | None = None,
    existing_content: list[dict[str, str]] | None = None,
    revision_instruction: str = "",
) -> list[dict[str, str]]:
    """Сгенерировать сценарий через GenAPI/OpenAI-compatible endpoint."""
    if not AI_API_KEY:
        return make_script(topic, duration, tone)

    duration_label = "Shorts/Reels на 60 секунд" if duration == "short" else "YouTube-видео на 5-10 минут"
    tone_label = {
        "hype": "динамичный хайп, быстро, энергично, без воды",
        "emotional": "эмоциональный, искренний, мотивирующий",
        "educational": "обучающий, практичный, пошаговый",
    }[tone]
    options = options or {}
    option_labels = {
        "platform": "Площадка",
        "audience": "Целевая аудитория",
        "goal": "Цель видео",
        "format": "Формат съёмки",
        "cta": "Призыв к действию",
    }
    brief_lines = [
        f"{option_labels[key]}: {value}"
        for key, value in options.items()
        if key in option_labels and value
    ]
    brief = "\n".join(brief_lines) or "Дополнительных пожеланий нет"
    revision_context = ""
    if existing_content and revision_instruction:
        revision_context = (
            "\n\nЭто улучшение уже существующего сценария.\n"
            f"Текущий сценарий: {json.dumps(existing_content, ensure_ascii=False)}\n"
            f"Задача улучшения: {revision_instruction}\n"
            "Сохрани удачные детали и структуру остальных блоков."
        )

    system_prompt = (
        "Ты сильный русскоязычный сценарист коротких видео и режиссер монтажа. "
        "Пиши конкретно, применимо и без общих фраз. "
        "Не объясняй формат, верни только валидный JSON."
    )
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
- Верни ровно JSON-массив из 3 блоков.

Схема каждого блока:
{{
  "title": "название блока",
  "timing": "тайминг",
  "frame": "что происходит в кадре",
  "speaker": "что говорить спикеру",
  "light": "световая атмосфера",
  "sound": "звуки и эффекты"
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
        "max_tokens": 1800 if duration == "short" else 3200,
    }

    timeout = ClientTimeout(total=AI_TIMEOUT_SECONDS)
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
        data = await poll_genapi_result(request_id)
        answer = extract_ai_message_content(data)

    try:
        parsed = extract_json_from_ai_answer(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        logging.error("AI response is not valid JSON: %s", error)
        raise RuntimeError("ИИ вернула ответ не в JSON-формате") from error

    if not isinstance(parsed, list) or len(parsed) < 3:
        logging.error("AI response has invalid scene list")
        raise RuntimeError("ИИ вернула неправильную структуру сценария")

    scenes = [normalize_ai_scene(scene, index, duration) for index, scene in enumerate(parsed[:3])]
    if any(not scene["speaker"] or not scene["frame"] for scene in scenes):
        logging.error("AI response has empty required scene fields")
        raise RuntimeError("ИИ вернула неполный сценарий")
    return scenes


def make_script(topic: str, duration: str, tone: str) -> list[dict[str, str]]:
    """Сформировать готовый структурированный сценарий без внешнего API."""
    is_short = duration == "short"
    duration_label = "60 секунд" if is_short else "5–10 минут"
    opening_timing = "00:00–00:04" if is_short else "00:00–00:25"
    body_timing = "00:05–00:42" if is_short else "00:26–06:30"
    finale_timing = "00:43–01:00" if is_short else "06:31–08:00"

    tone_copy = {
        "hype": "говорим быстро, уверенно и с нарастающей энергией",
        "emotional": "говорим искренне, через личное переживание и короткие паузы",
        "educational": "объясняем просто, точно и по шагам",
    }[tone]

    return [
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

    try:
        content = await generate_ai_script(
            scenario["topic"],
            scenario["duration"],
            requested_tone,
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
        content = await generate_ai_script(topic, duration, tone, options=options)
    except RuntimeError as error:
        return web.json_response({"error": str(error)}, status=502)
    scenario_id = await asyncio.to_thread(
        database.save_scenario, user["id"], topic, duration, tone, content
    )
    return web.json_response(
        {
            "scenario": {
                "id": scenario_id,
                "topic": topic,
                "duration": duration,
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
