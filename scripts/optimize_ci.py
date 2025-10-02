#!/usr/bin/env python3
"""
CI/CD Performance Optimization Script

This script analyzes and optimizes CI/CD performance by:
1. Analyzing test execution times
2. Identifying slow tests
3. Suggesting optimizations
4. Generating performance reports
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


class CIPerformanceAnalyzer:
    """Analyzes and optimizes CI/CD performance."""

    def __init__(self):
        self.results: Dict = {}

    def analyze_test_performance(self) -> Dict:
        """Analyze test execution performance."""
        print("🔍 Analyzing test performance...")

        try:
            # Run tests with timing information
            result = subprocess.run(
                [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/",
                    "--durations=20",
                    "--tb=no",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"⚠️  Tests failed: {result.stderr}")
                return {"error": "Tests failed to run"}

            # Parse durations from output
            durations = []
            for line in result.stdout.split("\n"):
                if "slowest" in line or "durations" in line:
                    continue
                if "s" in line and "test_" in line:
                    try:
                        # Extract duration and test name
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            duration_str = parts[0]
                            test_name = " ".join(parts[1:])
                            duration = float(duration_str.replace("s", ""))
                            durations.append((duration, test_name))
                    except (ValueError, IndexError):
                        continue

            # Sort by duration (slowest first)
            durations.sort(reverse=True)

            return {
                "total_tests": len(durations),
                "slowest_tests": durations[:10],
                "average_duration": sum(d[0] for d in durations) / len(durations)
                if durations
                else 0,
                "total_duration": sum(d[0] for d in durations),
            }

        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out"}
        except Exception as e:
            return {"error": f"Failed to analyze tests: {str(e)}"}

    def analyze_coverage_performance(self) -> Dict:
        """Analyze test coverage performance."""
        print("📊 Analyzing coverage performance...")

        try:
            # Run coverage analysis
            result = subprocess.run(
                [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/",
                    "--cov=src/character_ai",
                    "--cov-report=json",
                    "--cov-report=xml",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return {"error": "Coverage analysis failed"}

            # Parse coverage data
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)

                return {
                    "total_lines": coverage_data.get("totals", {}).get(
                        "num_statements", 0
                    ),
                    "covered_lines": coverage_data.get("totals", {}).get(
                        "covered_lines", 0
                    ),
                    "coverage_percentage": coverage_data.get("totals", {}).get(
                        "percent_covered", 0
                    ),
                    "files_analyzed": len(coverage_data.get("files", {})),
                }
            else:
                return {"error": "Coverage file not found"}

        except Exception as e:
            return {"error": f"Failed to analyze coverage: {str(e)}"}

    def analyze_docker_performance(self) -> Dict:
        """Analyze Docker build performance."""
        print("🐳 Analyzing Docker build performance...")

        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                return {"error": "Docker not available"}

            # Analyze Dockerfile
            dockerfile = Path("Dockerfile")
            if not dockerfile.exists():
                return {"error": "Dockerfile not found"}

            with open(dockerfile, "r") as f:
                dockerfile_content = f.read()

            # Count layers and analyze structure
            lines = dockerfile_content.split("\n")
            run_commands = [line for line in lines if line.strip().startswith("RUN")]
            copy_commands = [line for line in lines if line.strip().startswith("COPY")]

            return {
                "total_lines": len(lines),
                "run_commands": len(run_commands),
                "copy_commands": len(copy_commands),
                "estimated_layers": len(
                    [
                        line
                        for line in lines
                        if line.strip() and not line.strip().startswith("#")
                    ]
                ),
            }

        except Exception as e:
            return {"error": f"Failed to analyze Docker: {str(e)}"}

    def generate_optimization_recommendations(
        self, test_data: Dict, coverage_data: Dict, docker_data: Dict
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Test performance recommendations
        if "slowest_tests" in test_data and test_data["slowest_tests"]:
            slow_tests = test_data["slowest_tests"][:5]
            if any(duration > 10 for duration, _ in slow_tests):
                recommendations.append("🐌 Consider optimizing slow tests (>10s):")
                for duration, test_name in slow_tests:
                    if duration > 10:
                        recommendations.append(f"   - {test_name}: {duration:.2f}s")

        # Coverage recommendations
        if "coverage_percentage" in coverage_data:
            coverage = coverage_data["coverage_percentage"]
            if coverage < 80:
                recommendations.append(
                    f"📊 Test coverage is {coverage:.1f}% - consider adding more tests"
                )
            elif coverage > 95:
                recommendations.append(f"📊 Excellent test coverage: {coverage:.1f}%")

        # Docker recommendations
        if "run_commands" in docker_data:
            run_commands = docker_data["run_commands"]
            if run_commands > 10:
                recommendations.append(
                    f"🐳 Consider combining RUN commands (currently {run_commands})"
                )

        # General recommendations
        recommendations.extend(
            [
                "⚡ Use parallel test execution with pytest-xdist",
                "💾 Enable Poetry dependency caching in CI",
                "🔄 Use incremental builds for Docker images",
                "📦 Consider splitting tests into fast/slow categories",
            ]
        )

        return recommendations

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        print("📈 Generating performance report...")

        # Analyze different aspects
        test_data = self.analyze_test_performance()
        coverage_data = self.analyze_coverage_performance()
        docker_data = self.analyze_docker_performance()

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(
            test_data, coverage_data, docker_data
        )

        # Compile report
        report = {
            "timestamp": time.time(),
            "test_performance": test_data,
            "coverage_performance": coverage_data,
            "docker_performance": docker_data,
            "recommendations": recommendations,
            "summary": {
                "test_count": test_data.get("total_tests", 0),
                "average_test_duration": test_data.get("average_duration", 0),
                "coverage_percentage": coverage_data.get("coverage_percentage", 0),
                "docker_layers": docker_data.get("estimated_layers", 0),
            },
        }

        return report

    def save_report(
        self, report: Dict, output_file: str = "ci_performance_report.json"
    ):
        """Save performance report to file."""
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Performance report saved to: {output_file}")

    def print_summary(self, report: Dict):
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("🚀 CI/CD Performance Analysis Summary")
        print("=" * 60)

        summary = report["summary"]
        print(f"📊 Test Count: {summary['test_count']}")
        print(f"⏱️  Average Test Duration: {summary['average_test_duration']:.2f}s")
        print(f"📈 Coverage: {summary['coverage_percentage']:.1f}%")
        print(f"🐳 Docker Layers: {summary['docker_layers']}")

        print("\n💡 Optimization Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    analyzer = CIPerformanceAnalyzer()

    try:
        print("🔍 Starting CI/CD Performance Analysis...")

        # Generate report
        report = analyzer.generate_performance_report()

        # Save report
        analyzer.save_report(report)

        # Print summary
        analyzer.print_summary(report)

        print("\n🎉 Performance analysis complete!")

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
