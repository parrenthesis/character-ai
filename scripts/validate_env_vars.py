#!/usr/bin/env python3
"""
Environment Variable Validation Script

This script validates that all required environment variables are properly set
and have valid values for the Character AI platform.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class EnvVarRule:
    """Rule for validating an environment variable."""
    name: str
    required: bool
    description: str
    pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    example: Optional[str] = None


class EnvironmentValidator:
    """Validates environment variables for Character AI."""
    
    def __init__(self):
        self.rules = self._define_validation_rules()
        self.results: List[Dict] = []
        
    def _define_validation_rules(self) -> List[EnvVarRule]:
        """Define validation rules for all environment variables."""
        return [
            # Core Configuration
            EnvVarRule(
                name="CAI_ENVIRONMENT",
                required=True,
                description="Application environment (development, staging, production)",
                choices=["development", "staging", "production"],
                example="production"
            ),
            EnvVarRule(
                name="CAI_PATHS__MODELS_DIR",
                required=True,
                description="Directory path for AI models",
                pattern=r"^/.*",
                example="/app/models"
            ),
            EnvVarRule(
                name="CAI_MODELS__LLAMA_BACKEND",
                required=True,
                description="LLM backend type",
                choices=["llama_cpp", "transformers", "openai", "anthropic"],
                example="llama_cpp"
            ),
            
            # Security Configuration
            EnvVarRule(
                name="CAI_JWT_SECRET",
                required=True,
                description="JWT signing secret (should be long and random)",
                min_length=32,
                example="your-super-secure-jwt-secret-here-32-chars-min"
            ),
            EnvVarRule(
                name="CAI_PRIVATE_KEY_FILE",
                required=False,
                description="Path to private key file for JWT signing",
                pattern=r"^/.*\.pem$",
                example="/etc/cai/keys/private.pem"
            ),
            EnvVarRule(
                name="CAI_PUBLIC_KEY_FILE",
                required=False,
                description="Path to public key file for JWT verification",
                pattern=r"^/.*\.pem$",
                example="/etc/cai/keys/public.pem"
            ),
            EnvVarRule(
                name="CAI_REQUIRE_HTTPS",
                required=False,
                description="Require HTTPS connections",
                choices=["true", "false"],
                example="true"
            ),
            
            # Rate Limiting
            EnvVarRule(
                name="CAI_RATE_LIMIT_REQUESTS_PER_MINUTE",
                required=False,
                description="Rate limit for requests per minute",
                pattern=r"^\d+$",
                example="1000"
            ),
            EnvVarRule(
                name="CAI_RATE_LIMIT_BURST",
                required=False,
                description="Burst allowance for rate limiting",
                pattern=r"^\d+$",
                example="100"
            ),
            
            # Device Management
            EnvVarRule(
                name="CAI_ENABLE_DEVICE_REGISTRATION",
                required=False,
                description="Enable device registration",
                choices=["true", "false"],
                example="true"
            ),
            EnvVarRule(
                name="CAI_JWT_EXPIRY_SECONDS",
                required=False,
                description="JWT token expiry time in seconds",
                pattern=r"^\d+$",
                example="3600"
            ),
            EnvVarRule(
                name="CAI_JWT_ALGORITHM",
                required=False,
                description="JWT signing algorithm",
                choices=["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"],
                example="HS256"
            ),
            
            # API Configuration
            EnvVarRule(
                name="CAI_BASE_URL",
                required=False,
                description="Base URL for the API",
                pattern=r"^https?://.*",
                example="https://api.character.ai"
            ),
            EnvVarRule(
                name="CAI_HOST",
                required=False,
                description="Host to bind the API server",
                example="0.0.0.0"
            ),
            EnvVarRule(
                name="CAI_PORT",
                required=False,
                description="Port for the API server",
                pattern=r"^\d+$",
                example="8000"
            ),
            
            # Monitoring
            EnvVarRule(
                name="CAI_ENABLE_MONITORING",
                required=False,
                description="Enable monitoring and metrics",
                choices=["true", "false"],
                example="true"
            ),
            EnvVarRule(
                name="CAI_PROMETHEUS_PORT",
                required=False,
                description="Port for Prometheus metrics",
                pattern=r"^\d+$",
                example="9090"
            ),
            
            # Logging
            EnvVarRule(
                name="CAI_LOG_LEVEL",
                required=False,
                description="Logging level",
                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                example="INFO"
            ),
            EnvVarRule(
                name="CAI_LOG_FORMAT",
                required=False,
                description="Log format (json, text)",
                choices=["json", "text"],
                example="json"
            ),
            
            # Database (if applicable)
            EnvVarRule(
                name="CAI_DATABASE_URL",
                required=False,
                description="Database connection URL",
                pattern=r"^.*://.*",
                example="postgresql://user:pass@localhost:5432/character_ai"
            ),
            
            # External Services
            EnvVarRule(
                name="CAI_OPENAI_API_KEY",
                required=False,
                description="OpenAI API key",
                min_length=20,
                example="sk-..."
            ),
            EnvVarRule(
                name="CAI_ANTHROPIC_API_KEY",
                required=False,
                description="Anthropic API key",
                min_length=20,
                example="sk-ant-..."
            ),
            EnvVarRule(
                name="CAI_OLLAMA_BASE_URL",
                required=False,
                description="Ollama base URL",
                pattern=r"^https?://.*",
                example="http://localhost:11434"
            ),
        ]
    
    def validate_single_var(self, rule: EnvVarRule) -> Dict[str, Any]:
        """Validate a single environment variable."""
        value = os.getenv(rule.name)
        
        result = {
            "name": rule.name,
            "required": rule.required,
            "description": rule.description,
            "value": value,
            "is_set": value is not None,
            "is_valid": True,
            "errors": []
        }
        
        # Check if required variable is missing
        if rule.required and not value:
            result["is_valid"] = False
            result["errors"].append("Required environment variable is not set")
            return result
        
        # Skip validation if not set and not required
        if not value:
            return result
        
        # Validate pattern
        if rule.pattern and not re.match(rule.pattern, value):
            result["is_valid"] = False
            result["errors"].append(f"Value does not match required pattern: {rule.pattern}")
        
        # Validate choices
        if rule.choices and value not in rule.choices:
            result["is_valid"] = False
            result["errors"].append(f"Value must be one of: {', '.join(rule.choices)}")
        
        # Validate length
        if rule.min_length and len(value) < rule.min_length:
            result["is_valid"] = False
            result["errors"].append(f"Value must be at least {rule.min_length} characters long")
        
        if rule.max_length and len(value) > rule.max_length:
            result["is_valid"] = False
            result["errors"].append(f"Value must be no more than {rule.max_length} characters long")
        
        return result
    
    def validate_all_vars(self) -> Dict[str, Any]:
        """Validate all environment variables."""
        results = []
        
        for rule in self.rules:
            result = self.validate_single_var(rule)
            results.append(result)
        
        # Calculate summary
        total_vars = len(results)
        required_vars = sum(1 for r in results if r["required"])
        set_vars = sum(1 for r in results if r["is_set"])
        valid_vars = sum(1 for r in results if r["is_valid"])
        required_set = sum(1 for r in results if r["required"] and r["is_set"])
        required_valid = sum(1 for r in results if r["required"] and r["is_valid"])
        
        return {
            "summary": {
                "total_variables": total_vars,
                "required_variables": required_vars,
                "set_variables": set_vars,
                "valid_variables": valid_vars,
                "required_set": required_set,
                "required_valid": required_valid,
                "completion_rate": (set_vars / total_vars * 100) if total_vars > 0 else 0,
                "required_completion_rate": (required_set / required_vars * 100) if required_vars > 0 else 0,
                "validity_rate": (valid_vars / total_vars * 100) if total_vars > 0 else 0
            },
            "results": results
        }
    
    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable report."""
        summary = validation_results["summary"]
        results = validation_results["results"]
        
        report = []
        report.append("🔍 Environment Variable Validation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("📊 Summary:")
        report.append(f"  Total Variables: {summary['total_variables']}")
        report.append(f"  Required Variables: {summary['required_variables']}")
        report.append(f"  Set Variables: {summary['set_variables']} ({summary['completion_rate']:.1f}%)")
        report.append(f"  Valid Variables: {summary['valid_variables']} ({summary['validity_rate']:.1f}%)")
        report.append(f"  Required Set: {summary['required_set']}/{summary['required_variables']} ({summary['required_completion_rate']:.1f}%)")
        report.append("")
        
        # Required variables status
        report.append("🔑 Required Variables:")
        required_results = [r for r in results if r["required"]]
        for result in required_results:
            status = "✅" if result["is_valid"] else "❌"
            set_status = "SET" if result["is_set"] else "NOT SET"
            report.append(f"  {status} {result['name']} ({set_status})")
            if result["errors"]:
                for error in result["errors"]:
                    report.append(f"    ⚠️  {error}")
        report.append("")
        
        # Optional variables status
        report.append("🔧 Optional Variables:")
        optional_results = [r for r in results if not r["required"]]
        set_optional = [r for r in optional_results if r["is_set"]]
        
        if set_optional:
            for result in set_optional:
                status = "✅" if result["is_valid"] else "❌"
                report.append(f"  {status} {result['name']}")
                if result["errors"]:
                    for error in result["errors"]:
                        report.append(f"    ⚠️  {error}")
        else:
            report.append("  (No optional variables set)")
        report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        missing_required = [r for r in required_results if not r["is_set"]]
        if missing_required:
            report.append("  Required variables that need to be set:")
            for result in missing_required:
                report.append(f"    - {result['name']}: {result['description']}")
                if result.get("example"):
                    report.append(f"      Example: {result['example']}")
        else:
            report.append("  ✅ All required variables are set!")
        
        invalid_vars = [r for r in results if r["is_set"] and not r["is_valid"]]
        if invalid_vars:
            report.append("  Variables with invalid values:")
            for result in invalid_vars:
                report.append(f"    - {result['name']}: {', '.join(result['errors'])}")
        
        return "\n".join(report)
    
    def save_results(self, validation_results: Dict[str, Any], output_file: str = "env_validation_report.json"):
        """Save validation results to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"📄 Detailed results saved to: {output_file}")


def main():
    """Main entry point."""
    validator = EnvironmentValidator()
    
    try:
        print("🔍 Validating Environment Variables...")
        print("=" * 60)
        
        # Run validation
        results = validator.validate_all_vars()
        
        # Generate and print report
        report = validator.generate_report(results)
        print(report)
        
        # Save detailed results
        validator.save_results(results)
        
        # Check if validation passed
        summary = results["summary"]
        if summary["required_valid"] == summary["required_variables"]:
            print("\n🎉 All required environment variables are valid!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {summary['required_variables'] - summary['required_valid']} required variables are invalid or missing!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
