#!/usr/bin/env python3
"""
Restore script for memory system data.

Restores conversation history, user preferences, and other memory-related
data from backup files for disaster recovery and migration.
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


class MemoryDataRestore:
    """Handles restoration of memory system data."""

    def __init__(self, data_directory: str = "data"):
        """Initialize with data directory path."""
        self.data_directory = Path(data_directory)

    def restore_backup(self, backup_path: str, force: bool = False) -> Dict[str, Any]:
        """Restore memory data from backup."""
        backup_file = Path(backup_path)

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        restore_info = {
            "backup_file": str(backup_file),
            "data_directory": str(self.data_directory),
            "restore_timestamp": datetime.now().isoformat(),
            "files_restored": [],
            "errors": [],
        }

        try:
            # Create data directory if it doesn't exist
            self.data_directory.mkdir(parents=True, exist_ok=True)

            # Check for existing data
            existing_files = self._check_existing_data()
            if existing_files and not force:
                raise FileExistsError(
                    f"Data directory contains existing files: {existing_files}. "
                    "Use --force to overwrite."
                )

            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                # Get backup metadata if available
                metadata = self._extract_metadata(tar)
                if metadata:
                    restore_info["backup_metadata"] = metadata

                # Extract files
                for member in tar.getmembers():
                    if member.isfile() and member.name != "backup_metadata.json":
                        # Extract to data directory
                        member.name = Path(
                            member.name
                        ).name  # Remove any path components
                        tar.extract(member, self.data_directory)
                        restore_info["files_restored"].append(member.name)

            # Verify restored data
            verification_results = self._verify_restored_data()
            restore_info["verification"] = verification_results

            if not verification_results["success"]:
                restore_info["errors"].extend(verification_results["errors"])
                logger.warning("Data verification failed, but files were restored")

            logger.info(
                f"Restore completed: {len(restore_info['files_restored'])} files restored"
            )
            return restore_info

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            restore_info["errors"].append(str(e))
            raise

    def _check_existing_data(self) -> List[str]:
        """Check for existing data files."""
        existing_files = []

        # Check for key files
        key_files = ["conversations.db", "user_preferences.json"]
        for file_name in key_files:
            file_path = self.data_directory / file_name
            if file_path.exists():
                existing_files.append(file_name)

        return existing_files

    def _extract_metadata(self, tar: tarfile.TarFile) -> Dict[str, Any]:
        """Extract backup metadata if available."""
        try:
            metadata_member = tar.getmember("backup_metadata.json")
            metadata_file = tar.extractfile(metadata_member)
            if metadata_file:
                metadata_str = metadata_file.read().decode("utf-8")
                return json.loads(metadata_str)
        except KeyError:
            # No metadata file in backup
            pass
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return {}

    def _verify_restored_data(self) -> Dict[str, Any]:
        """Verify restored data integrity."""
        verification = {
            "success": True,
            "errors": [],
            "database_stats": {},
            "preferences_stats": {},
        }

        # Verify database
        db_path = self.data_directory / "conversations.db"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()

                    # Check database integrity
                    cursor.execute("PRAGMA integrity_check")
                    integrity_result = cursor.fetchone()[0]

                    if integrity_result != "ok":
                        verification["errors"].append(
                            f"Database integrity check failed: {integrity_result}"
                        )
                        verification["success"] = False

                    # Get basic stats
                    cursor.execute("SELECT COUNT(*) FROM conversations")
                    conversation_count = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                    unique_users = cursor.fetchone()[0]

                    verification["database_stats"] = {
                        "conversation_count": conversation_count,
                        "unique_users": unique_users,
                        "integrity_check": integrity_result,
                    }

            except Exception as e:
                verification["errors"].append(f"Database verification failed: {e}")
                verification["success"] = False
        else:
            verification["errors"].append("Database file not found after restore")
            verification["success"] = False

        # Verify preferences file
        prefs_path = self.data_directory / "user_preferences.json"
        if prefs_path.exists():
            try:
                with open(prefs_path, "r") as f:
                    data = json.load(f)

                verification["preferences_stats"] = {
                    "user_count": len(data),
                    "file_size_bytes": prefs_path.stat().st_size,
                }

            except Exception as e:
                verification["errors"].append(f"Preferences verification failed: {e}")
                verification["success"] = False
        else:
            verification["errors"].append("Preferences file not found after restore")
            verification["success"] = False

        return verification

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

    def preview_backup(self, backup_path: str) -> Dict[str, Any]:
        """Preview backup contents without extracting."""
        backup_file = Path(backup_path)

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        preview = {
            "backup_file": str(backup_file),
            "files": [],
            "metadata": {},
            "size_bytes": backup_file.stat().st_size,
        }

        try:
            with tarfile.open(backup_file, "r:gz") as tar:
                # List files
                for member in tar.getmembers():
                    if member.isfile():
                        preview["files"].append(
                            {
                                "name": member.name,
                                "size": member.size,
                                "mtime": datetime.fromtimestamp(member.mtime),
                            }
                        )

                # Extract metadata if available
                try:
                    metadata_member = tar.getmember("backup_metadata.json")
                    metadata_file = tar.extractfile(metadata_member)
                    if metadata_file:
                        metadata_str = metadata_file.read().decode("utf-8")
                        preview["metadata"] = json.loads(metadata_str)
                except KeyError:
                    pass

        except Exception as e:
            logger.error(f"Failed to preview backup: {e}")
            raise

        return preview


def main():
    """Main restore script entry point."""
    parser = argparse.ArgumentParser(description="Restore memory system data")
    parser.add_argument("backup_path", help="Path to backup file to restore")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory path (default: data)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force restore, overwriting existing data"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview backup contents without restoring",
    )
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    restore_manager = MemoryDataRestore(args.data_dir)

    if args.list:
        # List backups
        backups = restore_manager.list_backups("backups")
        if backups:
            print("Available backups:")
            for backup in backups:
                size_mb = backup["size_bytes"] / 1024 / 1024
                print(
                    f"  {backup['filename']} ({size_mb:.2f} MB) - {backup['created']}"
                )
        else:
            print("No backups found")

    elif args.preview:
        # Preview backup
        try:
            preview = restore_manager.preview_backup(args.backup_path)
            print(f"Backup preview: {args.backup_path}")
            print(f"Size: {preview['size_bytes'] / 1024 / 1024:.2f} MB")
            print(f"Files: {len(preview['files'])}")

            if preview["files"]:
                print("\nFiles in backup:")
                for file_info in preview["files"]:
                    print(f"  {file_info['name']} ({file_info['size']} bytes)")

            if preview["metadata"]:
                print("\nBackup metadata:")
                print(
                    f"  Created: {preview['metadata'].get('backup_timestamp', 'Unknown')}"
                )
                if "database_stats" in preview["metadata"]:
                    db_stats = preview["metadata"]["database_stats"]
                    print(f"  Conversations: {db_stats.get('conversation_count', 0)}")
                    print(f"  Users: {db_stats.get('unique_users', 0)}")

        except Exception as e:
            logger.error(f"Preview failed: {e}")
            exit(1)

    else:
        # Restore backup
        try:
            restore_info = restore_manager.restore_backup(args.backup_path, args.force)
            print(f"Restore completed successfully: {args.backup_path}")
            print(f"Files restored: {', '.join(restore_info['files_restored'])}")

            if restore_info.get("verification", {}).get("database_stats"):
                db_stats = restore_info["verification"]["database_stats"]
                print(
                    f"Database: {db_stats['conversation_count']} conversations, "
                    f"{db_stats['unique_users']} users"
                )

            if restore_info.get("verification", {}).get("preferences_stats"):
                prefs_stats = restore_info["verification"]["preferences_stats"]
                print(f"Preferences: {prefs_stats['user_count']} users")

            if restore_info["errors"]:
                print(f"Warnings: {', '.join(restore_info['errors'])}")

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            exit(1)


if __name__ == "__main__":
    main()
