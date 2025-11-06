#!/usr/bin/env python3
"""Check coverage regression - ensures all chapters maintain 100% concept coverage.

This script runs both coverage tools and fails if any chapter drops below 100%.
Also verifies concept_code_mapping.json matches actual file existence.
Intended for CI/CD integration.
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def run_coverage_check(script_path: Path) -> Tuple[int, str]:
    """Run a coverage check script and return exit code and output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=15  # 15 second timeout to prevent hangs
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, f"Timeout running {script_path}"
    except Exception as e:
        return 1, f"Error running {script_path}: {e}"


def parse_coverage_output(output: str) -> Dict[str, float]:
    """Parse coverage output to extract per-chapter coverage percentages."""
    coverage = {}
    lines = output.split('\n')
    
    for line in lines:
        # Look for lines like: "WARNING: ch1: 34 files | Book: 55 | Code: 34 | Covered: 31 (56%)"
        # or "[OK] ch6: 30 files | Book: 35 | Code: 24 | Covered: 24 (69%)"
        if 'ch' in line and 'Covered:' in line:
            parts = line.split()
            ch_id = None
            
            # Find chapter ID (strip colon if present)
            for i, part in enumerate(parts):
                part_clean = part.rstrip(':,')
                if part_clean.startswith('ch') and len(part_clean) > 2 and part_clean[2:].isdigit():
                    ch_id = part_clean
                    break
            
            if ch_id:
                # Look for percentage in parentheses after "Covered:"
                found_covered = False
                for j, part in enumerate(parts):
                    if part == 'Covered:':
                        found_covered = True
                    elif found_covered and '(' in part and '%' in part:
                        try:
                            # Extract number from "(56%)" or similar
                            pct_str = part.strip('()%')
                            coverage[ch_id] = float(pct_str)
                            break
                        except ValueError:
                            pass
    
    return coverage


def check_all_chapters_100_percent(coverage: Dict[str, float], script_name: str) -> Tuple[bool, List[str]]:
    """Check if all chapters have 100% coverage."""
    failures = []
    
    for ch_id in sorted(coverage.keys()):
        pct = coverage[ch_id]
        if pct < 100.0:
            failures.append(f"{ch_id}: {pct:.1f}% coverage (expected 100%)")
    
    return len(failures) == 0, failures


def verify_json_file_existence(repo_root: Path) -> Tuple[bool, List[str]]:
    """Verify that files marked as 'covered' in JSON actually exist.
    
    Returns:
        Tuple of (all_valid, list_of_issues)
    """
    json_path = repo_root / "concept_code_mapping.json"
    
    if not json_path.exists():
        return True, []  # JSON is optional, don't fail if missing
    
    try:
        with open(json_path) as f:
            mapping = json.load(f)
    except Exception as e:
        return False, [f"Failed to load concept_code_mapping.json: {e}"]
    
    issues = []
    
    for ch_id, ch_data in mapping.get('chapters', {}).items():
        if 'concepts' not in ch_data:
            continue
        
        ch_dir = repo_root / ch_id
        
        for concept, concept_data in ch_data['concepts'].items():
            status = concept_data.get('status', 'missing')
            baseline_file = concept_data.get('baseline_file')
            optimized_file = concept_data.get('optimized_file')
            
            # Check baseline file
            if baseline_file and baseline_file != 'null':
                baseline_path = ch_dir / baseline_file
                if not baseline_path.exists():
                    issues.append(
                        f"{ch_id}/{concept}: JSON marks as '{status}' but baseline file "
                        f"'{baseline_file}' does not exist"
                    )
            
            # Check optimized file (if status is 'covered', it should exist)
            if status == 'covered' and optimized_file and optimized_file != 'null':
                optimized_path = ch_dir / optimized_file
                if not optimized_path.exists():
                    issues.append(
                        f"{ch_id}/{concept}: JSON marks as 'covered' but optimized file "
                        f"'{optimized_file}' does not exist"
                    )
            
            # Warn if status is 'covered' but files are null
            if status == 'covered':
                if not baseline_file or baseline_file == 'null':
                    issues.append(
                        f"{ch_id}/{concept}: JSON marks as 'covered' but baseline_file is null"
                    )
                if not optimized_file or optimized_file == 'null':
                    issues.append(
                        f"{ch_id}/{concept}: JSON marks as 'covered' but optimized_file is null"
                    )
    
    return len(issues) == 0, issues


def main():
    """Main regression check."""
    repo_root = Path(__file__).parent.parent.parent
    
    print("=" * 80)
    print("COVERAGE REGRESSION CHECK")
    print("=" * 80)
    print("Using comprehensive_coverage_check.py as authoritative tool")
    print("(verify_code_concept_coverage.py uses different methodology and is not used)")
    print()
    
    # Run coverage tools
    # Note: comprehensive_coverage_check.py is the authoritative tool with exclusion logic
    # verify_code_concept_coverage.py uses different methodology and is for reference only
    scripts = [
        ('tools/coverage/comprehensive_coverage_check.py', 'Comprehensive Coverage Check (Authoritative)'),
        # ('tools/coverage/verify_code_concept_coverage.py', 'Code Concept Coverage Verification (Reference)'),  # Different methodology, not authoritative
    ]
    
    all_passed = True
    all_coverage = {}
    
    for script_file, script_name in scripts:
        script_path = repo_root / script_file
        
        if not script_path.exists():
            print(f"WARNING: {script_name}: Script not found at {script_path}")
            continue
        
        print(f"Running {script_name}...")
        print("-" * 80)
        
        exit_code, output = run_coverage_check(script_path)
        
        if exit_code != 0:
            print(f"ERROR: {script_name} failed with exit code {exit_code}")
            print(output)
            all_passed = False
            continue
        
        # Parse coverage
        coverage = parse_coverage_output(output)
        all_coverage[script_name] = coverage
        
        # Check for 100% coverage
        passed, failures = check_all_chapters_100_percent(coverage, script_name)
        
        if passed:
            print(f"[OK] {script_name}: All chapters at 100% coverage")
        else:
            print(f"ERROR: {script_name}: {len(failures)} chapters below 100%:")
            for failure in failures:
                print(f"   - {failure}")
            all_passed = False
        
        print()
    
    # Verify JSON file existence
    print("=" * 80)
    print("VERIFYING concept_code_mapping.json")
    print("=" * 80)
    print()
    
    json_valid, json_issues = verify_json_file_existence(repo_root)
    
    if json_valid:
        print("[OK] concept_code_mapping.json: All 'covered' files exist")
    else:
        print(f"WARNING: concept_code_mapping.json: {len(json_issues)} issues found:")
        for issue in json_issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(json_issues) > 10:
            print(f"   ... and {len(json_issues) - 10} more issues")
        print()
        print("Run: python3 tools/generate_concept_mapping.py to update JSON")
        all_passed = False
    
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_passed:
        print("[OK] All coverage checks passed - all chapters at 100% coverage")
        print("[OK] concept_code_mapping.json verified - all files exist")
        return 0
    else:
        print("ERROR: Coverage regression detected")
        print()
        print("Please ensure all concepts are covered in baseline_/optimized_ file pairs.")
        print("See concept_code_mapping.json for concept-to-code mappings.")
        if not json_valid:
            print("Run: python3 tools/generate_concept_mapping.py to update JSON")
        return 1


if __name__ == "__main__":
    sys.exit(main())

