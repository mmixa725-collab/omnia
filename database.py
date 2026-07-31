"""Слой хранения данных omnia на SQLite.

Модуль не зависит от aiogram и может использоваться отдельно. Каждая операция
открывает короткое соединение: это упрощает безопасную работу бота и HTTP API
в одном процессе.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_FREE_GENERATIONS = 3


class ClosingSQLiteConnection(sqlite3.Connection):
    """SQLite-соединение, которое гарантированно закрывается после блока with."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class Database:
    """Небольшой репозиторий пользователей, сценариев и платежей."""

    def __init__(self, database_path: str | Path = "omnia.db") -> None:
        self.database_path = str(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            factory=ClosingSQLiteConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        """Создать таблицы и индексы, если они ещё не существуют."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    free_generations INTEGER NOT NULL DEFAULT 3
                        CHECK (free_generations >= 0),
                    is_premium INTEGER NOT NULL DEFAULT 0
                        CHECK (is_premium IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    duration TEXT NOT NULL,
                    duration_minutes INTEGER,
                    tone TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS payments (
                    telegram_payment_charge_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    invoice_payload TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS scenarios_user_created_index
                ON scenarios(user_id, created_at DESC);
                """
            )
            scenario_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(scenarios)").fetchall()
            }
            if "is_favorite" not in scenario_columns:
                connection.execute(
                    "ALTER TABLE scenarios ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (is_favorite IN (0, 1))"
                )
            if "duration_minutes" not in scenario_columns:
                connection.execute(
                    "ALTER TABLE scenarios ADD COLUMN duration_minutes INTEGER"
                )

    @staticmethod
    def _user_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        premium = bool(row["is_premium"])
        return {
            "id": row["id"],
            "username": row["username"] or "Пользователь",
            "free_generations": row["free_generations"],
            "is_premium": premium,
            "generations_label": "Без лимита" if premium else str(row["free_generations"]),
        }

    def ensure_user(self, user_id: int, username: str | None) -> dict[str, Any]:
        """Создать пользователя с тремя генерациями либо обновить его имя."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, free_generations, is_premium)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    username = excluded.username,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, username, DEFAULT_FREE_GENERATIONS),
            )
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError("Не удалось создать пользователя")
        return self._user_to_dict(row)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._user_to_dict(row) if row else None

    def consume_generation(self, user_id: int) -> dict[str, Any] | None:
        """Атомарно списать одну генерацию.

        Для премиум-пользователя счётчик не меняется. Если бесплатный лимит
        исчерпан, возвращается None.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None

            if not row["is_premium"]:
                updated = connection.execute(
                    """
                    UPDATE users
                    SET free_generations = free_generations - 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND free_generations > 0
                    """,
                    (user_id,),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    return None

            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            connection.commit()
            return self._user_to_dict(row)
        finally:
            connection.close()

    def save_scenario(
        self,
        user_id: int,
        topic: str,
        duration: str,
        tone: str,
        content: list[dict[str, str]],
        duration_minutes: int | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scenarios
                    (user_id, topic, duration, duration_minutes, tone, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    topic,
                    duration,
                    duration_minutes,
                    tone,
                    json.dumps(content, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_scenarios(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, topic, duration, duration_minutes, tone, content, is_favorite, created_at
                FROM scenarios
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "topic": row["topic"],
                "duration": row["duration"],
                "duration_minutes": row["duration_minutes"],
                "tone": row["tone"],
                "content": json.loads(row["content"]),
                "is_favorite": bool(row["is_favorite"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_scenario(self, user_id: int, scenario_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, topic, duration, duration_minutes, tone, content, is_favorite, created_at
                FROM scenarios WHERE id = ? AND user_id = ?
                """,
                (scenario_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "topic": row["topic"],
            "duration": row["duration"],
            "duration_minutes": row["duration_minutes"],
            "tone": row["tone"],
            "content": json.loads(row["content"]),
            "is_favorite": bool(row["is_favorite"]),
            "created_at": row["created_at"],
        }

    def update_scenario(
        self,
        user_id: int,
        scenario_id: int,
        *,
        topic: str | None = None,
        tone: str | None = None,
        content: list[dict[str, str]] | None = None,
        is_favorite: bool | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        values: list[Any] = []
        if topic is not None:
            updates.append("topic = ?")
            values.append(topic)
        if tone is not None:
            updates.append("tone = ?")
            values.append(tone)
        if content is not None:
            updates.append("content = ?")
            values.append(json.dumps(content, ensure_ascii=False))
        if is_favorite is not None:
            updates.append("is_favorite = ?")
            values.append(int(is_favorite))
        if updates:
            with self._connect() as connection:
                connection.execute(
                    f"UPDATE scenarios SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                    (*values, scenario_id, user_id),
                )
        return self.get_scenario(user_id, scenario_id)

    def delete_scenario(self, user_id: int, scenario_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM scenarios WHERE id = ? AND user_id = ?",
                (scenario_id, user_id),
            )
            return cursor.rowcount == 1

    def activate_premium(
        self,
        user_id: int,
        payment_charge_id: str,
        invoice_payload: str,
        amount: int,
        currency: str,
    ) -> bool:
        """Сохранить платёж и включить premium; повторный update безопасен."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO payments
                    (telegram_payment_charge_id, user_id, invoice_payload, amount, currency)
                VALUES (?, ?, ?, ?, ?)
                """,
                (payment_charge_id, user_id, invoice_payload, amount, currency),
            )
            connection.execute(
                """
                UPDATE users
                SET is_premium = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user_id,),
            )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()


class PostgresDatabase:
    """PostgreSQL-хранилище для бесплатного облачного запуска.

    Интерфейс совпадает с Database, поэтому локально приложение продолжает
    использовать SQLite, а на хостинге переключается одной переменной.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as error:
            raise RuntimeError(
                "Для PostgreSQL установите зависимости из requirements.txt"
            ) from error
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _user_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        premium = bool(row["is_premium"])
        return {
            "id": row["id"],
            "username": row["username"] or "Пользователь",
            "free_generations": row["free_generations"],
            "is_premium": premium,
            "generations_label": "Без лимита" if premium else str(row["free_generations"]),
        }

    def initialize(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS users (
                id BIGINT PRIMARY KEY,
                username TEXT,
                free_generations INTEGER NOT NULL DEFAULT 3 CHECK (free_generations >= 0),
                is_premium INTEGER NOT NULL DEFAULT 0 CHECK (is_premium IN (0, 1)),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                topic TEXT NOT NULL,
                duration TEXT NOT NULL,
                duration_minutes INTEGER,
                tone TEXT NOT NULL,
                content JSONB NOT NULL,
                is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1)),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS payments (
                telegram_payment_charge_id TEXT PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                invoice_payload TEXT NOT NULL,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS scenarios_user_created_index ON scenarios(user_id, created_at DESC)",
            "ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1))",
            "ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS duration_minutes INTEGER",
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def ensure_user(self, user_id: int, username: str | None) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (id, username, free_generations, is_premium)
                    VALUES (%s, %s, %s, 0)
                    ON CONFLICT(id) DO UPDATE SET
                        username = excluded.username,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                    """,
                    (user_id, username, DEFAULT_FREE_GENERATIONS),
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Не удалось создать пользователя")
        return self._user_to_dict(row)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cursor.fetchone()
        return self._user_to_dict(row) if row else None

    def consume_generation(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = %s FOR UPDATE", (user_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                if not row["is_premium"]:
                    cursor.execute(
                        """
                        UPDATE users
                        SET free_generations = free_generations - 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND free_generations > 0
                        RETURNING *
                        """,
                        (user_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                return self._user_to_dict(row)

    def save_scenario(
        self,
        user_id: int,
        topic: str,
        duration: str,
        tone: str,
        content: list[dict[str, str]],
        duration_minutes: int | None = None,
    ) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scenarios
                        (user_id, topic, duration, duration_minutes, tone, content)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        user_id,
                        topic,
                        duration,
                        duration_minutes,
                        tone,
                        json.dumps(content, ensure_ascii=False),
                    ),
                )
                row = cursor.fetchone()
                return int(row["id"])

    def list_scenarios(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, topic, duration, duration_minutes, tone, content, is_favorite, created_at
                    FROM scenarios WHERE user_id = %s
                    ORDER BY id DESC LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "topic": row["topic"],
                "duration": row["duration"],
                "duration_minutes": row["duration_minutes"],
                "tone": row["tone"],
                "content": row["content"],
                "is_favorite": bool(row["is_favorite"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]

    def get_scenario(self, user_id: int, scenario_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, topic, duration, duration_minutes, tone, content, is_favorite, created_at
                    FROM scenarios WHERE id = %s AND user_id = %s
                    """,
                    (scenario_id, user_id),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "topic": row["topic"],
            "duration": row["duration"],
            "duration_minutes": row["duration_minutes"],
            "tone": row["tone"],
            "content": row["content"],
            "is_favorite": bool(row["is_favorite"]),
            "created_at": row["created_at"].isoformat(),
        }

    def update_scenario(
        self,
        user_id: int,
        scenario_id: int,
        *,
        topic: str | None = None,
        tone: str | None = None,
        content: list[dict[str, str]] | None = None,
        is_favorite: bool | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        values: list[Any] = []
        if topic is not None:
            updates.append("topic = %s")
            values.append(topic)
        if tone is not None:
            updates.append("tone = %s")
            values.append(tone)
        if content is not None:
            updates.append("content = %s::jsonb")
            values.append(json.dumps(content, ensure_ascii=False))
        if is_favorite is not None:
            updates.append("is_favorite = %s")
            values.append(int(is_favorite))
        if updates:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"UPDATE scenarios SET {', '.join(updates)} WHERE id = %s AND user_id = %s",
                        (*values, scenario_id, user_id),
                    )
        return self.get_scenario(user_id, scenario_id)

    def delete_scenario(self, user_id: int, scenario_id: int) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM scenarios WHERE id = %s AND user_id = %s",
                    (scenario_id, user_id),
                )
                return cursor.rowcount == 1

    def activate_premium(
        self,
        user_id: int,
        payment_charge_id: str,
        invoice_payload: str,
        amount: int,
        currency: str,
    ) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO payments
                        (telegram_payment_charge_id, user_id, invoice_payload, amount, currency)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(telegram_payment_charge_id) DO NOTHING
                    """,
                    (payment_charge_id, user_id, invoice_payload, amount, currency),
                )
                inserted = cursor.rowcount == 1
                cursor.execute(
                    """
                    UPDATE users SET is_premium = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (user_id,),
                )
                return inserted


def create_database(database_location: str):
    """Выбрать PostgreSQL в облаке или SQLite при локальном запуске."""
    if database_location.startswith(("postgres://", "postgresql://")):
        return PostgresDatabase(database_location)
    return Database(database_location)


if __name__ == "__main__":
    database = Database()
    database.initialize()
    print("База данных omnia готова: omnia.db")
