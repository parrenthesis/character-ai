#!/usr/bin/env python3
"""
Backup script for memory system data.

Creates compressed backups of conversation history, user preferences,
and other memory-related data for disaster recovery and migration.
"""

import argparse
import json
import logging
import sqlite3
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MemoryDataBackup:
    """Handles backup of memory system data."""

    def __init__(self, data_directory: str = "data"):
        """Initialize with data directory path."""
        self.data_directory = Path(data_directory)
        self.backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_backup(self, output_path: str) -> Dict[str, Any]:
        """Create a complete backup of memory data."""
        backup_path = Path(output_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        backup_info = {
            "timestamp": self.backup_timestamp,
            "data_directory": str(self.data_directory),
            "files_backed_up": [],
            "database_stats": {},
            "preferences_stats": {},
            "backup_size": 0,
        }

        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                # Backup conversation database
                db_path = self.data_directory / "conversations.db"
                if db_path.exists():
                    tar.add(db_path, arcname="conversations.db")
                    backup_info["files_backed_up"].append("conversations.db")
                    backup_info["database_stats"] = self._get_database_stats(db_path)

                # Backup user preferences
                prefs_path = self.data_directory / "user_preferences.json"
                if prefs_path.exists():
                    tar.add(prefs_path, arcname="user_preferences.json")
                    backup_info["files_backed_up"].append("user_preferences.json")
                    backup_info["preferences_stats"] = self._get_preferences_stats(
                        prefs_path
                    )

                # Backup any other memory-related files
                for file_path in self.data_directory.glob("*.json"):
                    if file_path.name not in ["user_preferences.json"]:
                        tar.add(file_path, arcname=file_path.name)
                        backup_info["files_backed_up"].append(file_path.name)

                # Add backup metadata
                metadata = {
                    "backup_timestamp": self.backup_timestamp,
                    "data_directory": str(self.data_directory),
                    "files": backup_info["files_backed_up"],
                    "database_stats": backup_info["database_stats"],
                    "preferences_stats": backup_info["preferences_stats"],
                }

                # Create metadata file
                metadata_str = json.dumps(metadata, indent=2)
                tar.addfile(
                    tarfile.TarInfo("backup_metadata.json"),
                    fileobj=type("", (), {"read": lambda: metadata_str.encode()})(),
                )
                backup_info["files_backed_up"].append("backup_metadata.json")

            # Get backup size
            backup_info["backup_size"] = backup_path.stat().st_size

            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(
                f"Backup size: {backup_info['backup_size'] / 1024 / 1024:.2f} MB"
            )

            return backup_info

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def _get_database_stats(self, db_path: Path) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Count conversations
                cursor.execute("SELECT COUNT(*) FROM conversations")
                conversation_count = cursor.fetchone()[0]

                # Count sessions
                cursor.execute("SELECT COUNT(*) FROM sessions")
                session_count = cursor.fetchone()[0]

                # Count unique users
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                unique_users = cursor.fetchone()[0]

                # Database size
                db_size = db_path.stat().st_size

                return {
                    "conversation_count": conversation_count,
                    "session_count": session_count,
                    "unique_users": unique_users,
                    "database_size_bytes": db_size,
                }

        except Exception as e:
            logger.warning(f"Failed to get database stats: {e}")
            return {}

    def _get_preferences_stats(self, prefs_path: Path) -> Dict[str, Any]:
        """Get preferences file statistics."""
        try:
            with open(prefs_path, "r") as f:
                data = json.load(f)

            return {
                "user_count": len(data),
                "file_size_bytes": prefs_path.stat().st_size,
            }

        except Exception as e:
            logger.warning(f"Failed to get preferences stats: {e}")
            return {}

    def list_backups(self, backup_directory: str) -> List[Dict[str, Any]]:
        """List available backups."""
        backup_dir = Path(backup_directory)
        backups = []

        for backup_file in backup_dir.glob("memory_backup_*.tar.gz"):
            try:
                stat = backup_file.stat()
                backups.append(
                    {
                        "filename": backup_file.name,
                        "size_bytes": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime),
                        "path": str(backup_file),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to process backup file {backup_file}: {e}")

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def verify_backup(self, backup_path: str) -> bool:
        """Verify backup integrity."""
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                # Check if essential files are present
                essential_files = ["conversations.db", "user_preferences.json"]
                files_in_backup = tar.getnames()

                missing_files = [f for f in essential_files if f not in files_in_backup]
                if missing_files:
                    logger.warning(
                        f"Missing essential files in backup: {missing_files}"
                    )
                    return False

                # Try to extract and verify database
                for member in tar.getmembers():
                    if member.name == "conversations.db":
                        db_file = tar.extractfile(member)
                        if db_file:
                            # Basic SQLite header check
                            header = db_file.read(16)
                            if not header.startswith(b"SQLite format 3"):
                                logger.error("Invalid SQLite database in backup")
                                return False

                logger.info("Backup verification successful")
                return True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False


def main():
    """Main backup script entry point."""
    parser = argparse.ArgumentParser(description="Backup memory system data")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory path (default: data)"
    )
    parser.add_argument(
        "--output",
        default=f"backups/memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz",
        help="Output backup file path",
    )
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--verify", help="Verify backup file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    backup_manager = MemoryDataBackup(args.data_dir)

    if args.list:
        # List backups
        backups = backup_manager.list_backups("backups")
        if backups:
            print("Available backups:")
            for backup in backups:
                size_mb = backup["size_bytes"] / 1024 / 1024
                print(
                    f"  {backup['filename']} ({size_mb:.2f} MB) - {backup['created']}"
                )
        else:
            print("No backups found")

    elif args.verify:
        # Verify backup
        if backup_manager.verify_backup(args.verify):
            print(f"Backup verification successful: {args.verify}")
        else:
            print(f"Backup verification failed: {args.verify}")
            exit(1)

    else:
        # Create backup
        try:
            backup_info = backup_manager.create_backup(args.output)
            print(f"Backup created successfully: {args.output}")
            print(f"Files backed up: {', '.join(backup_info['files_backed_up'])}")
            print(f"Backup size: {backup_info['backup_size'] / 1024 / 1024:.2f} MB")

            if backup_info["database_stats"]:
                print(
                    f"Database: {backup_info['database_stats']['conversation_count']} conversations, "
                    f"{backup_info['database_stats']['unique_users']} users"
                )

            if backup_info["preferences_stats"]:
                print(
                    f"Preferences: {backup_info['preferences_stats']['user_count']} users"
                )

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            exit(1)


if __name__ == "__main__":
    main()
