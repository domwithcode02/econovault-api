#!/usr/bin/env python3
"""
EconoVault API Security Check Script

This script checks for common security issues in the repository:
1. Sensitive files that might be committed
2. API keys or secrets in code
3. Git history for accidentally committed secrets
4. File permissions issues
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

class SecurityChecker:
    def __init__(self, repo_root: str = None):
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.issues = []
        self.warnings = []
        
        # Common secret patterns
        self.secret_patterns = [
            r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'secret[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'password\s*[:=]\s*["\']?[a-zA-Z0-9]{8,}["\']?',
            r'token\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'auth[_-]?token\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'private[_-]?key\s*[:=]\s*["\']?-----BEGIN[A-Z\s]+PRIVATE KEY-----',
            r'aws[_-]?(access|secret)[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'github[_-]?token\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
            r'bearer\s+["\']?[a-zA-Z0-9]{20,}["\']?',
            r'basic\s+[a-zA-Z0-9+/=]{20,}',
        ]
        
        # Files that should never be committed
        self.sensitive_files = [
            '.env',
            '.env.local',
            '.env.development.local',
            '.env.test.local',
            '.env.production.local',
            'environments/dev.env',
            'environments/prod.env',
            'environments/staging.env',
            'config/local.py',
            'config/production.py',
            'secrets/',
            'keys/',
            '*.key',
            '*.pem',
            '*.p12',
            '*.pfx',
        ]
        
        # File extensions that might contain secrets
        self.sensitive_extensions = [
            '.key',
            '.pem',
            '.p12',
            '.pfx',
            '.crt',
            '.csr',
        ]

    def run_check(self) -> bool:
        """Run all security checks"""
        print("EconoVault API Security Check")
        print("=" * 50)
        
        all_passed = True
        
        # Check 1: Sensitive files in repository
        print("\nChecking for sensitive files...")
        all_passed &= self._check_sensitive_files()
        
        # Check 2: Secrets in code
        print("\nScanning for secrets in code...")
        all_passed &= self._scan_for_secrets()
        
        # Check 3: Git history for secrets
        print("\nChecking git history for secrets...")
        all_passed &= self._check_git_history()
        
        # Check 4: .gitignore effectiveness
        print("\nChecking .gitignore coverage...")
        all_passed &= self._check_gitignore()
        
        # Check 5: File permissions
        print("\nChecking file permissions...")
        all_passed &= self._check_file_permissions()
        
        # Summary
        print("\n" + "=" * 50)
        print("SECURITY CHECK SUMMARY")
        print("=" * 50)
        
        if self.issues:
            print("CRITICAL ISSUES FOUND:")
            for issue in self.issues:
                print(f"   • {issue}")
            all_passed = False
        
        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if all_passed:
            print("All security checks passed!")
            print("Repository appears to be secure.")
        else:
            print("Security issues found. Please address them before committing.")
        
        return all_passed

    def _check_sensitive_files(self) -> bool:
        """Check for sensitive files that shouldn't be in the repository"""
        passed = True
        
        for pattern in self.sensitive_files:
            if '*' in pattern:
                # Glob pattern
                for file_path in self.repo_root.glob(pattern):
                    if file_path.is_file():
                        # Check if it's a template or contains only placeholders
                        if self._is_safe_template_file(file_path):
                            self.warnings.append(f"Template file found (should be .gitignored in production): {file_path.relative_to(self.repo_root)}")
                        else:
                            self.issues.append(f"Sensitive file found: {file_path.relative_to(self.repo_root)}")
                            passed = False
            else:
                # Exact path
                file_path = self.repo_root / pattern
                if file_path.exists() and file_path.is_file():
                    # Check if it's a template or contains only placeholders
                    if self._is_safe_template_file(file_path):
                        self.warnings.append(f"Template file found (should be .gitignored in production): {pattern}")
                    else:
                        self.issues.append(f"Sensitive file found: {pattern}")
                        passed = False
        
        if passed:
            print("No sensitive files found in repository")
        
        return passed

    def _is_safe_template_file(self, file_path: Path) -> bool:
        """Check if a file is a safe template with only placeholders"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file contains only placeholders and comments
            lines = content.split('\n')
            secret_fields_found = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check if it's a variable assignment
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    # Only consider it a secret if the key indicates it's sensitive
                    # More specific secret field patterns to avoid false positives
                    secret_key_patterns = [
                        r'_?key$', r'_?secret$', r'_?password$', r'_?token$', 
                        r'_?api_?key$', r'_?encryption_?key$', r'_?auth_?token$',
                        r'^secret_', r'^api_?key_', r'^password_', r'^token_'
                    ]
                    import re
                    is_secret_field = any(re.search(pattern, key.lower()) for pattern in secret_key_patterns)
                    
                    if is_secret_field and value and not self._is_placeholder(value):
                        secret_fields_found.append(f"{key}={value}")
            
            return len(secret_fields_found) == 0
            
        except (UnicodeDecodeError, PermissionError):
            return False

    def _scan_for_secrets(self) -> bool:
        """Scan code files for potential secrets"""
        passed = True
        scanned_files = 0
        
        # File patterns to scan
        scan_patterns = [
            '**/*.py',
            '**/*.js',
            '**/*.ts',
            '**/*.json',
            '**/*.yaml',
            '**/*.yml',
            '**/*.md',
            '**/*.txt',
        ]
        
        # Exclude patterns
        exclude_patterns = [
            '.git',
            'node_modules',
            '__pycache__',
            '.venv',
            'venv',
            'env',
            'dist',
            'build',
            '.opencode',
        ]
        
        for pattern in scan_patterns:
            for file_path in self.repo_root.glob(pattern):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in exclude_patterns):
                    continue
                
                # Skip files that should be ignored
                if file_path.name.startswith('.'):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        scanned_files += 1
                        
                        # Check each secret pattern
                        for pattern in self.secret_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                line_content = content.split('\n')[line_num - 1].strip()
                                
                                # Skip if it's just a placeholder or example
                                if self._is_placeholder(line_content):
                                    continue
                                
                                relative_path = file_path.relative_to(self.repo_root)
                                self.issues.append(
                                    f"Potential secret found in {relative_path}:{line_num}: {line_content[:50]}..."
                                )
                                passed = False
                
                except (UnicodeDecodeError, PermissionError):
                    # Skip binary files or files we can't read
                    continue
        
        print(f"Scanned {scanned_files} files for secrets")
        return passed

    def _check_git_history(self) -> bool:
        """Check git history for accidentally committed secrets"""
        passed = True
        
        try:
            # Check if we're in a git repository
            if not (self.repo_root / '.git').exists():
                self.warnings.append("Not a git repository, skipping history check")
                return True
            
            # Check for secrets in commit history
            result = subprocess.run(
                ['git', 'log', '--name-status', '--pretty=format:%H'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                # Check for sensitive file patterns in history
                for line in result.stdout.split('\n'):
                    for pattern in self.sensitive_files:
                        if pattern.replace('*', '') in line:
                            self.warnings.append(f"Sensitive file pattern found in git history: {line}")
                            passed = False
            
            # Check for secrets in commit messages
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%s'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                for commit_msg in result.stdout.split('\n'):
                    for pattern in self.secret_patterns:
                        if re.search(pattern, commit_msg, re.IGNORECASE):
                            self.issues.append(f"Potential secret in commit message: {commit_msg[:50]}...")
                            passed = False
        
        except subprocess.CalledProcessError:
            self.warnings.append("Git command failed, skipping history check")
        except FileNotFoundError:
            self.warnings.append("Git not found, skipping history check")
        
        if passed:
            print("No secrets found in git history")
        
        return passed

    def _check_gitignore(self) -> bool:
        """Check if .gitignore covers all necessary patterns"""
        passed = True
        
        gitignore_path = self.repo_root / '.gitignore'
        if not gitignore_path.exists():
            self.issues.append(".gitignore file not found")
            return False
        
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        # Check for essential patterns
        essential_patterns = [
            '.env',
            '.env.local',
            '*.key',
            '*.pem',
            'secrets/',
            'keys/',
            'environments/*.env',
        ]
        
        missing_patterns = []
        for pattern in essential_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            self.warnings.append(f".gitignore missing recommended patterns: {', '.join(missing_patterns)}")
            passed = False
        else:
            print(".gitignore covers essential patterns")
        
        return passed

    def _check_file_permissions(self) -> bool:
        """Check file permissions for sensitive files"""
        passed = True
        
        # Check for world-readable sensitive files
        sensitive_patterns = [
            '.env*',
            '*.key',
            '*.pem',
            'environments/*.env',
        ]
        
        for pattern in sensitive_patterns:
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file():
                    try:
                        # Check if file is world-readable
                        stat = file_path.stat()
                        if stat.st_mode & 0o044:  # World-readable
                            self.warnings.append(f"World-readable file: {file_path.relative_to(self.repo_root)}")
                            passed = False
                    except OSError:
                        # Skip files we can't access
                        continue
        
        if passed:
            print("File permissions look good")
        
        return passed

    def _is_placeholder(self, text: str) -> bool:
        """Check if text is a placeholder rather than a real secret"""
        placeholder_indicators = [
            'your-', 'placeholder', 'example', 'test-', 'demo-', 'fake-',
            'set_your_', 'your_api_key', 'your_secret', 'your_token',
            'change_me', 'replace_me', 'enter_your', 'set_your_'
        ]
        
        # Specific placeholder values we've set
        specific_placeholders = [
            'set_your_bls_api_key_here',
            'set_your_bea_api_key_here', 
            'set_your_fred_api_key_here',
            'your-secret-key-here-min-32-characters',
            'your-encryption-key-here-min-32-characters',
            'your-bls-api-key-here',
            'your-bea-api-key-here',
            'your-fred-api-key-here',
            'testpass_placeholder'
        ]
        
        # Variable substitution patterns (safe)
        variable_patterns = [
            '${',  # Environment variable substitution
            '$(',   # Subshell substitution
        ]
        
        text_lower = text.lower().strip()
        original_text = text.strip()
        
        # Check specific placeholders first
        if text_lower in specific_placeholders:
            return True
            
        # Check variable substitution patterns (safe)
        if any(pattern in original_text for pattern in variable_patterns):
            return True
            
        # Check general patterns
        return any(indicator in text_lower for indicator in placeholder_indicators)

def main():
    """Main function"""
    checker = SecurityChecker()
    success = checker.run_check()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()