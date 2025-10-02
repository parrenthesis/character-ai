"""
CLI commands for LLM management.

Provides command-line interface for managing LLMs, models, and configurations.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, List, Optional

from .config import LLMConfigManager
from .config import LLMProvider as ConfigLLMProvider
from .config import LLMType
from .factory import LLMFactory
from .manager import ModelInfo, OpenModelManager

logger = logging.getLogger(__name__)


class LLMCLI:
    """CLI interface for LLM management."""

    def __init__(self) -> None:
        self.config_manager = LLMConfigManager()
        self.model_manager = OpenModelManager()
        self.factory = LLMFactory(self.config_manager, self.model_manager)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for LLM CLI."""
        parser = argparse.ArgumentParser(
            description="Character AI - LLM Management CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  cai llm list-models                    # List available models
  cai llm install llama-3.2-1b-instruct  # Install a model
  cai llm status                         # Show LLM status
  cai llm config --character-creation-provider local  # Configure LLM
  cai llm test                          # Test LLM connections
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Model management commands
        self._add_model_commands(subparsers)

        # Configuration commands
        self._add_config_commands(subparsers)

        # Status and testing commands
        self._add_status_commands(subparsers)

        return parser

    def _add_model_commands(self, subparsers: Any) -> None:
        """Add model management commands."""
        # List models
        list_parser = subparsers.add_parser("list-models", help="List available models")

        list_parser.add_argument(
            "--installed", action="store_true", help="Show only installed models"
        )
        list_parser.add_argument(
            "--use-case",
            choices=["character_creation", "runtime"],
            help="Filter by use case",
        )

        # Install model
        install_parser = subparsers.add_parser("install", help="Install a model")
        install_parser.add_argument("model_name", help="Name of model to install")
        install_parser.add_argument(
            "--progress", action="store_true", help="Show download progress"
        )

        # Remove model
        remove_parser = subparsers.add_parser("remove", help="Remove a model")
        remove_parser.add_argument("model_name", help="Name of model to remove")

        # Model info
        info_parser = subparsers.add_parser("model-info", help="Show model information")

        info_parser.add_argument("model_name", help="Name of model")

    def _add_config_commands(self, subparsers: Any) -> None:
        """Add configuration commands."""
        config_parser = subparsers.add_parser("config", help="Configure LLM settings")

        # Character creation config
        config_parser.add_argument(
            "--character-creation-provider",
            choices=["local", "ollama", "openai", "anthropic"],
            help="Set character creation provider",
        )
        config_parser.add_argument(
            "--character-creation-model", help="Set character creation model"
        )

        # Runtime config
        config_parser.add_argument(
            "--runtime-provider",
            choices=["local", "ollama", "openai", "anthropic"],
            help="Set runtime provider",
        )
        config_parser.add_argument("--runtime-model", help="Set runtime model")

        # Provider configs
        config_parser.add_argument("--local-model-path", help="Set local model path")
        config_parser.add_argument("--ollama-base-url", help="Set Ollama base URL")

        # Save/load config
        config_parser.add_argument("--save", help="Save configuration to file")
        config_parser.add_argument("--load", help="Load configuration from file")
        config_parser.add_argument(
            "--show", action="store_true", help="Show current configuration"
        )

    def _add_status_commands(self, subparsers: Any) -> None:
        """Add status and testing commands."""
        # Status
        status_parser = subparsers.add_parser("status", help="Show LLM status")
        status_parser.add_argument("--json", action="store_true", help="Output as JSON")

        # Test
        test_parser = subparsers.add_parser("test", help="Test LLM connections")
        test_parser.add_argument(
            "--llm-type",
            choices=["character_creation", "runtime", "both"],
            default="both",
            help="Which LLM to test",
        )

        # Storage
        storage_parser = subparsers.add_parser("storage", help="Show storage usage")
        storage_parser.add_argument(
            "--cleanup", action="store_true", help="Clean up orphaned files"
        )

    async def run(self, args: Optional[List[str]] = None) -> None:
        """Run CLI with arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return

        try:
            if parsed_args.command == "list-models":
                await self._handle_list_models(parsed_args)
            elif parsed_args.command == "install":
                await self._handle_install_model(parsed_args)
            elif parsed_args.command == "remove":
                await self._handle_remove_model(parsed_args)
            elif parsed_args.command == "model-info":
                await self._handle_model_info(parsed_args)
            elif parsed_args.command == "config":
                await self._handle_config(parsed_args)
            elif parsed_args.command == "status":
                await self._handle_status(parsed_args)
            elif parsed_args.command == "test":
                await self._handle_test(parsed_args)
            elif parsed_args.command == "storage":
                await self._handle_storage(parsed_args)
            else:
                parser.print_help()
        except Exception as e:
            logger.error(f"Command failed: {e}")
            sys.exit(1)

    async def _handle_list_models(self, args: Any) -> None:
        """Handle list-models command."""
        if args.installed:
            installed_models = self.model_manager.list_installed_models()
            print("Installed models:")
            for model in installed_models:
                print(f"  - {model}")
        else:
            available_models: List[
                ModelInfo
            ] = self.model_manager.list_available_models()
            if args.use_case:
                available_models = [
                    m
                    for m in available_models
                    if m.recommended_for and args.use_case in m.recommended_for
                ]

            print("Available models:")
            for model in available_models:  # type: ignore
                installed = (
                    "✓" if self.model_manager.is_model_installed(model.name) else " "  # type: ignore
                )
                print(
                    f"  {installed} {model.name} ({model.size}) - {model.description}"  # type: ignore
                )
                print(f"    Recommended for: {', '.join(model.recommended_for or [])}")  # type: ignore

    async def _handle_install_model(self, args: Any) -> None:
        """Handle install command."""
        model_name = args.model_name

        if self.model_manager.is_model_installed(model_name):
            print(f"Model {model_name} is already installed")
            return

        model_info = self.model_manager.get_model_info(model_name)
        if not model_info:
            print(f"Unknown model: {model_name}")
            return

        print(f"Installing {model_name} ({model_info.size})...")

        def progress_callback(progress: float) -> None:
            if args.progress:
                print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        success = await self.model_manager.download_model(model_name, progress_callback)

        if success:
            print(f"\n✓ Successfully installed {model_name}")
        else:
            print(f"\n✗ Failed to install {model_name}")

    async def _handle_remove_model(self, args: Any) -> None:
        """Handle remove command."""
        model_name = args.model_name

        if not self.model_manager.is_model_installed(model_name):
            print(f"Model {model_name} is not installed")
            return

        success = self.model_manager.remove_model(model_name)
        if success:
            print(f"✓ Removed {model_name}")
        else:
            print(f"✗ Failed to remove {model_name}")

    async def _handle_model_info(self, args: Any) -> None:
        """Handle model-info command."""
        model_name = args.model_name
        model_info = self.model_manager.get_model_info(model_name)

        if not model_info:
            print(f"Unknown model: {model_name}")
            return

        print(f"Model: {model_info.name}")
        print(f"Size: {model_info.size}")
        print(f"Description: {model_info.description}")
        print(f"Format: {model_info.format}")
        print(f"Quantization: {model_info.quantization}")
        print(f"Recommended for: {', '.join(model_info.recommended_for or [])}")

        if self.model_manager.is_model_installed(model_name):
            model_path = self.model_manager.get_model_path(model_name)
            print(f"Installed at: {model_path}")
        else:
            print("Status: Not installed")

    async def _handle_config(self, args: Any) -> None:
        """Handle config command."""
        if args.show:
            self._show_config()
            return

        if args.load:
            self.config_manager.load_config(args.load)
            print(f"Configuration loaded from {args.load}")
            return

        # Update configuration
        if args.character_creation_provider:
            self.config_manager.character_creation.provider = ConfigLLMProvider(
                args.character_creation_provider
            )
            print(
                f"Character creation provider set to {args.character_creation_provider}"
            )

        if args.character_creation_model:
            self.config_manager.character_creation.model = args.character_creation_model

            print(f"Character creation model set to {args.character_creation_model}")

        if args.runtime_provider:
            self.config_manager.runtime.provider = ConfigLLMProvider(
                args.runtime_provider
            )
            print(f"Runtime provider set to {args.runtime_provider}")

        if args.runtime_model:
            self.config_manager.runtime.model = args.runtime_model
            print(f"Runtime model set to {args.runtime_model}")

        if args.local_model_path:
            self.config_manager.providers.local_model_path = args.local_model_path
            print(f"Local model path set to {args.local_model_path}")

        if args.ollama_base_url:
            self.config_manager.providers.ollama_base_url = args.ollama_base_url
            print(f"Ollama base URL set to {args.ollama_base_url}")

        # Save configuration
        if args.save:
            self.config_manager.save_config(args.save)
            print(f"Configuration saved to {args.save}")

    def _show_config(self) -> None:
        """Show current configuration."""
        print("Current LLM Configuration:")
        print(
            f"Character Creation: {self.config_manager.character_creation.provider.value} - {self.config_manager.character_creation.model}"
        )
        print(
            f"Runtime: {self.config_manager.runtime.provider.value} - {self.config_manager.runtime.model}"
        )
        print(f"Local Model Path: {self.config_manager.providers.local_model_path}")
        print(f"Ollama Base URL: {self.config_manager.providers.ollama_base_url}")
        print(
            f"Cost Tracking: {'Enabled' if self.config_manager.cost_tracking_enabled else 'Disabled'}"
        )
        print(
            f"Fallback: {'Enabled' if self.config_manager.fallback_enabled else 'Disabled'}"
        )

    async def _handle_status(self, args: Any) -> None:
        """Handle status command."""
        status = self.factory.get_llm_status()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("LLM Status:")
            for llm_type, info in status.items():
                print(f"\n{llm_type.replace('_', ' ').title()}:")
                if "error" in info:
                    print(f"  Error: {info['error']}")
                else:
                    print(f"  Provider: {info['provider']}")
                    print(f"  Connection: {'✓' if info['connection_ok'] else '✗'}")
                    print(f"  Cost per token: ${info['cost_per_token']}")
                    print(
                        f"  Requires internet: {'Yes' if info['requires_internet'] else 'No'}"
                    )

    async def _handle_test(self, args: Any) -> None:
        """Handle test command."""
        if args.llm_type in ["character_creation", "both"]:
            print("Testing character creation LLM...")
            success = self.factory.test_llm_connection(LLMType.CHARACTER_CREATION)
            print(f"Character creation: {'✓' if success else '✗'}")

        if args.llm_type in ["runtime", "both"]:
            print("Testing runtime LLM...")
            success = self.factory.test_llm_connection(LLMType.RUNTIME)
            print(f"Runtime: {'✓' if success else '✗'}")

    async def _handle_storage(self, args: Any) -> None:
        """Handle storage command."""
        usage = self.model_manager.get_storage_usage()

        print("Storage Usage:")
        print(f"  Total size: {usage['total_size_gb']:.2f} GB")
        print(f"  Model count: {usage['model_count']}")
        print(f"  Models directory: {usage['models_dir']}")

        if args.cleanup:
            print("\nCleaning up orphaned files...")
            cleaned = self.model_manager.cleanup_orphaned_files()
            if cleaned:
                print(f"Cleaned up: {', '.join(cleaned)}")
            else:
                print("No orphaned files found")


def main() -> None:
    """Main CLI entry point."""
    cli = LLMCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
