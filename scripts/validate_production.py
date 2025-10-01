#!/usr/bin/env python3
"""
Production Environment Validation Script

This script validates that the production environment is properly configured
and all services are running correctly.
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class ProductionValidator:
    """Validates production environment configuration and health."""
    
    def __init__(self):
        self.base_url = os.getenv("CAI_BASE_URL", "http://localhost:8000")
        self.timeout = int(os.getenv("CAI_TIMEOUT", "30"))
        self.results: List[Dict] = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log a test result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.time(),
            "details": details or {}
        }
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        required_vars = [
            "CAI_ENVIRONMENT",
            "CAI_PATHS__MODELS_DIR",
            "CAI_MODELS__LLAMA_BACKEND"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.log_result(
                "Environment Variables",
                False,
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            return False
        
        self.log_result(
            "Environment Variables",
            True,
            "All required environment variables are set"
        )
        return True
    
    def check_docker_health(self) -> bool:
        """Check Docker container health."""
        try:
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.log_result(
                    "Docker Health",
                    False,
                    f"Docker is not running or accessible: {result.stderr}"
                )
                return False
            
            # Check for character-ai container
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        container = json.loads(line)
                        containers.append(container)
                    except json.JSONDecodeError:
                        continue
            
            character_ai_containers = [
                c for c in containers 
                if 'character-ai' in c.get('Names', '') or 'character-ai' in c.get('Image', '')
            ]
            
            if not character_ai_containers:
                self.log_result(
                    "Docker Health",
                    False,
                    "No character-ai containers found running"
                )
                return False
            
            self.log_result(
                "Docker Health",
                True,
                f"Found {len(character_ai_containers)} character-ai container(s) running"
            )
            return True
            
        except subprocess.TimeoutExpired:
            self.log_result(
                "Docker Health",
                False,
                "Docker command timed out"
            )
            return False
        except Exception as e:
            self.log_result(
                "Docker Health",
                False,
                f"Docker health check failed: {str(e)}"
            )
            return False
    
    def check_api_health(self) -> bool:
        """Check API health endpoints."""
        try:
            # Check main health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            
            if response.status_code != 200:
                self.log_result(
                    "API Health",
                    False,
                    f"Health endpoint returned status {response.status_code}"
                )
                return False
            
            health_data = response.json()
            if not health_data.get("status") == "healthy":
                self.log_result(
                    "API Health",
                    False,
                    f"API reports unhealthy status: {health_data}"
                )
                return False
            
            self.log_result(
                "API Health",
                True,
                "API health endpoint is responding correctly"
            )
            return True
            
        except requests.exceptions.RequestException as e:
            self.log_result(
                "API Health",
                False,
                f"API health check failed: {str(e)}"
            )
            return False
    
    def check_monitoring_services(self) -> bool:
        """Check monitoring services (Prometheus, Grafana)."""
        services = [
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3000/api/health")
        ]
        
        all_healthy = True
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_result(
                        f"{service_name} Health",
                        True,
                        f"{service_name} is responding correctly"
                    )
                else:
                    self.log_result(
                        f"{service_name} Health",
                        False,
                        f"{service_name} returned status {response.status_code}"
                    )
                    all_healthy = False
            except requests.exceptions.RequestException as e:
                self.log_result(
                    f"{service_name} Health",
                    False,
                    f"{service_name} health check failed: {str(e)}"
                )
                all_healthy = False
        
        return all_healthy
    
    def check_model_files(self) -> bool:
        """Check that required model files exist."""
        models_dir = Path(os.getenv("CAI_PATHS__MODELS_DIR", "/app/models"))
        required_files = [
            "llm/tinyllama-1.1b-q4_k_m.gguf",
            "whisper/base.pt",
            "whisper/tiny.pt"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = models_dir / file_path
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            self.log_result(
                "Model Files",
                False,
                f"Missing required model files: {', '.join(missing_files)}"
            )
            return False
        
        self.log_result(
            "Model Files",
            True,
            "All required model files are present"
        )
        return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.log_result(
                    "Disk Space",
                    False,
                    f"Failed to check disk space: {result.stderr}"
                )
                return False
            
            # Parse df output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                self.log_result(
                    "Disk Space",
                    False,
                    "Could not parse disk space information"
                )
                return False
            
            # Get available space (4th column)
            parts = lines[1].split()
            if len(parts) < 4:
                self.log_result(
                    "Disk Space",
                    False,
                    "Could not parse disk space information"
                )
                return False
            
            available = parts[3]
            self.log_result(
                "Disk Space",
                True,
                f"Available disk space: {available}"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Disk Space",
                False,
                f"Disk space check failed: {str(e)}"
            )
            return False
    
    def run_smoke_tests(self) -> bool:
        """Run basic smoke tests on the API."""
        smoke_tests = [
            ("GET /health", "GET", "/health"),
            ("GET /api/v1/toy/health", "GET", "/api/v1/toy/health"),
            ("GET /api/v1/characters", "GET", "/api/v1/characters"),
        ]
        
        all_passed = True
        for test_name, method, endpoint in smoke_tests:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                else:
                    response = requests.request(method, f"{self.base_url}{endpoint}", timeout=self.timeout)
                
                if response.status_code in [200, 201, 202]:
                    self.log_result(
                        f"Smoke Test: {test_name}",
                        True,
                        f"Status {response.status_code}"
                    )
                else:
                    self.log_result(
                        f"Smoke Test: {test_name}",
                        False,
                        f"Unexpected status {response.status_code}"
                    )
                    all_passed = False
                    
            except requests.exceptions.RequestException as e:
                self.log_result(
                    f"Smoke Test: {test_name}",
                    False,
                    f"Request failed: {str(e)}"
                )
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive validation report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "timestamp": time.time()
            },
            "results": self.results
        }
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("🔍 Starting Production Environment Validation...")
        print("=" * 60)
        
        checks = [
            self.check_environment_variables,
            self.check_docker_health,
            self.check_api_health,
            self.check_monitoring_services,
            self.check_model_files,
            self.check_disk_space,
            self.run_smoke_tests
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.log_result(
                    check.__name__,
                    False,
                    f"Check failed with exception: {str(e)}"
                )
                all_passed = False
        
        print("=" * 60)
        return all_passed


def main():
    """Main entry point."""
    validator = ProductionValidator()
    
    try:
        success = validator.run_all_checks()
        report = validator.generate_report()
        
        # Save report to file
        report_path = Path("production_validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Validation Report saved to: {report_path}")
        print(f"📈 Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if success:
            print("\n🎉 All production validation checks passed!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {report['summary']['failed']} validation checks failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
