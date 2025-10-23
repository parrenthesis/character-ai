"""
Memory system CLI commands.

Provides command-line access to memory system statistics, data export,
and operational monitoring for the hybrid memory system.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import click

from ..observability.memory_metrics import get_memory_metrics


@click.group()
def memory() -> None:
    """Memory system management commands."""
    pass


@memory.command()
@click.option("--data-dir", default="data", help="Data directory path")
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def stats(data_dir: str, format: str) -> None:
    """Show memory system statistics."""
    if format == "json":
        stats_data = _get_all_stats(data_dir)
        click.echo(json.dumps(stats_data, indent=2))
    else:
        _print_stats(data_dir)


@memory.command()
@click.argument("user_id")
@click.option("--data-dir", default="data", help="Data directory path")
def export_user(user_id: str, data_dir: str) -> None:
    """Export all data for a specific user (GDPR compliance)."""
    _export_user_data(user_id, data_dir)


@memory.command()
@click.option("--data-dir", default="data", help="Data directory path")
@click.option("--days", default=30, help="Age threshold in days")
def cleanup(data_dir: str, days: int) -> None:
    """Clean up old conversation data."""
    _cleanup_old_data(data_dir, days)


@memory.command()
@click.option("--data-dir", default="data", help="Data directory path")
def vacuum(data_dir: str) -> None:
    """Optimize database by running VACUUM."""
    _vacuum_database(data_dir)


@memory.command()
@click.option("--data-dir", default="data", help="Data directory path")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def clean(data_dir: str, confirm: bool) -> None:
    """Clean all memory data (conversations, preferences, summaries)."""
    data_path = Path(data_dir)

    if not data_path.exists():
        click.echo("✅ No data directory found - nothing to clean")
        return

    if not confirm:
        click.confirm(
            f"Are you sure you want to delete all data in {data_path}?", abort=True
        )

    try:
        import shutil

        shutil.rmtree(data_path)
        click.echo(f"✅ Cleaned all memory data from {data_path}")
    except Exception as e:
        click.echo(f"❌ Failed to clean data: {e}")


def _get_database_stats(db_path: str) -> Dict[str, Any]:
    """Get database statistics."""
    if not Path(db_path).exists():
        return {"error": "Database file not found"}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get table counts
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_turns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT character_name) FROM conversations")
            unique_characters = cursor.fetchone()[0]

            # Get recent activity
            cursor.execute(
                """
                SELECT character_name, COUNT(*) as turn_count
                FROM conversations
                WHERE timestamp > (strftime('%s', 'now') - 3600)
                GROUP BY character_name
                ORDER BY turn_count DESC
            """
            )
            recent_activity = cursor.fetchall()

            return {
                "total_turns": total_turns,
                "total_sessions": total_sessions,
                "unique_users": unique_users,
                "unique_characters": unique_characters,
                "recent_activity_1h": [
                    {"character": row[0], "turns": row[1]} for row in recent_activity
                ],
                "db_size_bytes": Path(db_path).stat().st_size,
            }
    except Exception as e:
        return {"error": f"Database error: {e}"}


def _get_preferences_stats(prefs_path: str) -> Dict[str, Any]:
    """Get preferences statistics."""
    if not Path(prefs_path).exists():
        return {"error": "Preferences file not found"}

    try:
        with open(prefs_path, "r") as f:
            data = json.load(f)

        total_users = len(data)
        total_preferences = 0

        for user_id, prefs in data.items():
            if isinstance(prefs, dict):
                # Count non-empty preference fields
                for field in ["name", "interests", "favorite_color", "dislikes"]:
                    if prefs.get(field):
                        if field in ["interests", "dislikes"]:
                            total_preferences += len(prefs[field])
                        else:
                            total_preferences += 1

        return {
            "total_users": total_users,
            "total_preferences": total_preferences,
            "file_size_bytes": Path(prefs_path).stat().st_size,
        }
    except Exception as e:
        return {"error": f"Preferences error: {e}"}


def _get_memory_system_stats() -> Dict[str, Any]:
    """Get memory system statistics."""
    try:
        metrics = get_memory_metrics()
        return metrics.get_metrics_summary()
    except Exception as e:
        return {"error": f"Metrics error: {e}"}


def _get_all_stats(data_dir: str) -> Dict[str, Any]:
    """Get all statistics."""
    return {
        "database": _get_database_stats(str(Path(data_dir) / "conversations.db")),
        "preferences": _get_preferences_stats(
            str(Path(data_dir) / "user_preferences.json")
        ),
        "system": _get_memory_system_stats(),
    }


def _format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format."""
    value = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _print_stats(data_dir: str) -> None:
    """Print memory system statistics."""
    click.echo("🧠 Character AI Memory System Statistics")
    click.echo("=" * 50)

    # Database stats
    db_path = Path(data_dir) / "conversations.db"
    click.echo(f"\n📊 Database Statistics ({db_path})")
    click.echo("-" * 30)
    db_stats = _get_database_stats(str(db_path))

    if "error" in db_stats:
        click.echo(f"❌ {db_stats['error']}")
    else:
        click.echo(f"Total turns: {db_stats['total_turns']}")
        click.echo(f"Total sessions: {db_stats['total_sessions']}")
        click.echo(f"Unique users: {db_stats['unique_users']}")
        click.echo(f"Unique characters: {db_stats['unique_characters']}")
        click.echo(f"Database size: {_format_bytes(db_stats['db_size_bytes'])}")

        if db_stats["recent_activity_1h"]:
            click.echo("\nRecent activity (1h):")
            for activity in db_stats["recent_activity_1h"]:
                click.echo(f"  {activity['character']}: {activity['turns']} turns")

    # Preferences stats
    prefs_path = Path(data_dir) / "user_preferences.json"
    click.echo(f"\n👤 Preferences Statistics ({prefs_path})")
    click.echo("-" * 30)
    prefs_stats = _get_preferences_stats(str(prefs_path))

    if "error" in prefs_stats:
        click.echo(f"❌ {prefs_stats['error']}")
    else:
        click.echo(f"Total users: {prefs_stats['total_users']}")
        click.echo(f"Total preferences: {prefs_stats['total_preferences']}")
        click.echo(f"File size: {_format_bytes(prefs_stats['file_size_bytes'])}")

    # Memory system stats
    click.echo("\n⚙️  Memory System Status")
    click.echo("-" * 30)
    system_stats = _get_memory_system_stats()

    if "error" in system_stats:
        click.echo(f"❌ {system_stats['error']}")
    else:
        click.echo(
            f"Metrics initialized: {system_stats.get('metrics_initialized', False)}"
        )
        click.echo(
            f"Components tracked: {', '.join(system_stats.get('components_tracked', []))}"
        )
        click.echo(
            f"Prometheus available: {system_stats.get('prometheus_available', False)}"
        )


def _export_user_data(user_id: str, data_dir: str) -> None:
    """Export user data for GDPR compliance."""
    click.echo(f"📤 Exporting data for user: {user_id}")
    click.echo("=" * 40)

    # Export from database
    db_path = Path(data_dir) / "conversations.db"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Get user sessions
                cursor.execute(
                    """
                    SELECT session_id, character_name, start_time, end_time, turn_count
                    FROM sessions
                    WHERE user_id = ?
                """,
                    (user_id,),
                )
                sessions = cursor.fetchall()

                # Get user conversations
                cursor.execute(
                    """
                    SELECT turn_id, timestamp, user_input, character_response, character_name
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp
                """,
                    (user_id,),
                )
                conversations = cursor.fetchall()

                export_data = {
                    "user_id": user_id,
                    "export_timestamp": str(Path().cwd()),
                    "sessions": [
                        {
                            "session_id": row[0],
                            "character_name": row[1],
                            "start_time": row[2],
                            "end_time": row[3],
                            "turn_count": row[4],
                        }
                        for row in sessions
                    ],
                    "conversations": [
                        {
                            "turn_id": row[0],
                            "timestamp": row[1],
                            "user_input": row[2],
                            "character_response": row[3],
                            "character_name": row[4],
                        }
                        for row in conversations
                    ],
                }

                click.echo(
                    f"Found {len(sessions)} sessions and {len(conversations)} conversations"
                )
                click.echo(f"Export data saved to: user_{user_id}_export.json")

                with open(f"user_{user_id}_export.json", "w") as f:
                    json.dump(export_data, f, indent=2)

        except Exception as e:
            click.echo(f"❌ Database export failed: {e}")

    # Export preferences
    prefs_path = Path(data_dir) / "user_preferences.json"
    if prefs_path.exists():
        try:
            with open(prefs_path, "r") as f:
                prefs_data = json.load(f)

            user_prefs = prefs_data.get(user_id, {})
            if user_prefs:
                click.echo("Found preferences for user")
                with open(f"user_{user_id}_preferences.json", "w") as f:
                    json.dump(user_prefs, f, indent=2)
            else:
                click.echo("No preferences found for user")

        except Exception as e:
            click.echo(f"❌ Preferences export failed: {e}")


def _cleanup_old_data(data_dir: str, days: int) -> None:
    """Clean up old conversation data."""
    db_path = Path(data_dir) / "conversations.db"
    if not db_path.exists():
        click.echo("❌ Database not found")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Count old records
            cursor.execute(
                """
                SELECT COUNT(*) FROM conversations
                WHERE timestamp < (strftime('%s', 'now') - ? * 24 * 60 * 60)
            """,
                (days,),
            )
            old_count = cursor.fetchone()[0]

            if old_count == 0:
                click.echo("✅ No old data to clean up")
                return

            # Delete old records
            cursor.execute(
                """
                DELETE FROM conversations
                WHERE timestamp < (strftime('%s', 'now') - ? * 24 * 60 * 60)
            """,
                (days,),
            )
            deleted_count = cursor.rowcount

            click.echo(f"✅ Cleaned up {deleted_count} old conversation records")

    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}")


def _vacuum_database(data_dir: str) -> None:
    """Optimize database by running VACUUM."""
    db_path = Path(data_dir) / "conversations.db"
    if not db_path.exists():
        click.echo("❌ Database not found")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get size before
            size_before = db_path.stat().st_size

            # Run VACUUM
            cursor.execute("VACUUM")

            # Get size after
            size_after = db_path.stat().st_size
            saved_bytes = size_before - size_after

            click.echo("✅ Database optimized")
            click.echo(f"Size before: {_format_bytes(size_before)}")
            click.echo(f"Size after: {_format_bytes(size_after)}")
            click.echo(f"Space saved: {_format_bytes(saved_bytes)}")

    except Exception as e:
        click.echo(f"❌ VACUUM failed: {e}")
