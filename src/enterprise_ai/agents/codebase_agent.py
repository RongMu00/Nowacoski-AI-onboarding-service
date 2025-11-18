import logging
from dataclasses import dataclass
from typing import Dict, List

import requests
from github import Github
from strands import Agent


@dataclass
class CodebaseOnboarding:
    """Represents an onboarding plan for a codebase"""
    repository_name: str
    main_technologies: List[str]
    setup_steps: List[str]
    key_components: List[Dict[str, str]]
    estimated_study_time: str

class CodebaseAgent:
    def __init__(self, github_token: str):
        """Initialize CodebaseAgent with GitHub token"""
        self.logger = logging.getLogger("codebase_agent")
        self.github = Github(github_token)
        self.agent = Agent(system_prompt=self._get_system_prompt())

    def _get_system_prompt(self) -> str:
        return """You are a codebase analysis assistant. For a given GitHub repository:
        1. Analyze the main technologies and frameworks used
        2. Create clear setup instructions for developers
        3. Identify key components and their purposes
        4. Suggest a learning path for new developers
        5. Estimate time needed to understand the codebase

        Format your response as a structured list with clear sections."""

    def create_onboarding_plan(self, repo_url: str) -> CodebaseOnboarding:
        """Create onboarding plan from GitHub repository"""
        try:
            # Extract repo information from URL
            repo_name = repo_url.split('github.com/')[-1]
            repo = self.github.get_repo(repo_name)

            # Collect repository information
            readme = self._get_readme(repo)
            main_files = self._get_main_files(repo)
            dependencies = self._get_dependencies(repo)

            # Generate analysis prompt
            prompt = f"""Analyze this repository:
            Repository: {repo.name}
            Description: {repo.description}
            README: {readme}
            Main files: {main_files}
            Dependencies: {dependencies}
            """

            # Get structured response from agent
            response = self.agent(prompt)

            # # Parse response into onboarding plan
            # return self._parse_response(repo.name, response)

            # 直接返回字符串，不要解析成复杂结构
            if hasattr(response, '__str__'):
                return str(response)
            else:
                return response

        except Exception as e:
            self.logger.error(f"Error analyzing repository: {e}")
            raise

    def _get_readme(self, repo) -> str:
        """Get repository README content"""
        try:
            readme = repo.get_readme()
            return readme.decoded_content.decode('utf-8')
        except:
            return "No README found"

    def _get_main_files(self, repo) -> List[str]:
        """Get main files from repository root"""
        return [content.name for content in repo.get_contents("")]

    def _get_dependencies(self, repo) -> Dict:
        """Get project dependencies"""
        try:
            # Try to find package.json, requirements.txt, etc.
            dependencies = {}
            for file_name in ['requirements.txt', 'package.json', 'pom.xml']:
                try:
                    content = repo.get_contents(file_name)
                    dependencies[file_name] = content.decoded_content.decode('utf-8')
                except:
                    continue
            return dependencies
        except:
            return {}

    def _parse_response(self, repo_name: str, response) -> CodebaseOnboarding:
        """Parse agent response into CodebaseOnboarding structure"""
        content = response.content.split('\n')
        return CodebaseOnboarding(
            repository_name=repo_name,
            main_technologies=self._extract_section(content, "Technologies"),
            setup_steps=self._extract_section(content, "Setup"),
            key_components=self._extract_components(content),
            estimated_study_time=self._extract_time_estimate(content)
        )

    def _extract_section(self, content: List[str], section_name: str) -> List[str]:
        """Helper to extract sections from response"""
        # Add extraction logic here
        return []
