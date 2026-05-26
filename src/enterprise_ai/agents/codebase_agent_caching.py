import logging
from typing import Dict, List

from github import Github
from strands import Agent

logger = logging.getLogger("codebase_agent")


class CodebaseAgent:
    """Pure worker agent for GitHub repository analysis.

    This agent only fetches from GitHub and generates analysis — it does NOT
    manage its own cache.  All cache-vs-fetch decisions are made by the
    orchestrator.
    """

    def __init__(self, github_token: str):
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

    def create_onboarding_plan(self, repo_url: str, query: str = None) -> str:
        """Fetch and analyze a GitHub repository.

        Always fetches from GitHub — the orchestrator decides whether to call
        this method or use cached content.

        Args:
            repo_url: GitHub repository URL
            query: Optional specific question about the repo.  When provided,
                   fetches targeted files from GitHub (via code search or tree
                   scan) and generates a focused answer instead of a generic
                   onboarding overview.
        """
        try:
            repo_name = repo_url.split('github.com/')[-1].strip('/')
            self.logger.info(f"📦 Processing repository: {repo_name}")

            repo = self.github.get_repo(repo_name)

            # Targeted mode: specific question → fetch relevant files
            if query:
                targeted = self._fetch_targeted_files(repo, query)
                if targeted:
                    return self._generate_targeted_answer(
                        repo_name, targeted, query
                    )
                # Fall through to general analysis if no targeted files found

            # General mode: full repo overview
            readme = self._get_readme(repo)
            main_files = self._get_main_files(repo)
            dependencies = self._get_dependencies(repo)
            languages = self._get_languages(repo)

            prompt = f"""Analyze this repository for new developer onboarding:

            Repository: {repo.name}
            Description: {repo.description}
            URL: {repo.html_url}
            Language: {', '.join(languages) if languages else 'Unknown'}

            Main Files: {', '.join(main_files[:10])}

            Dependencies/Package Info: {str(dependencies)[:500]}

            README Content (first 1000 chars):
            {readme[:1000] if readme else 'No README'}

            Provide structured onboarding plan including:
            1. Main Technologies Used
            2. Setup Instructions (step-by-step)
            3. Key Components (with brief descriptions)
            4. Learning Path for New Developers
            5. Estimated Time to Understand
            """

            response = self.agent(prompt)
            return self._format_response(str(response), repo_name)

        except Exception as e:
            self.logger.error(f"Error analyzing repository: {e}")
            return f"Error analyzing repository: {str(e)}"

    # ──────────────────────────────────────────────────────────
    # Targeted file fetching (worker logic — kept)
    # ──────────────────────────────────────────────────────────

    def _fetch_targeted_files(self, repo, query: str) -> List[Dict]:
        """Fetch specific files from GitHub that are relevant to the query.

        Uses GitHub code search to find files matching query keywords,
        then fetches their content. Returns list of {name, path, content}.
        """
        targeted = []

        try:
            import re as _re

            # Strip URLs from query before keyword extraction
            clean_query = _re.sub(r"https?://\S+", "", query)

            stop_words = {
                'what', 'does', 'the', 'how', 'is', 'in', 'a', 'an', 'for',
                'to', 'of', 'this', 'that', 'do', 'can', 'you', 'explain',
                'about', 'tell', 'me', 'file', 'code', 'function', 'class',
                'repo', 'repository', 'github', 'look', 'like', 'work',
                'implementation', 'entry', 'points',
            }
            # Keep only alphanumeric words, no special chars like ":"
            keywords = [
                w for w in _re.findall(r"[a-z0-9]+", clean_query.lower())
                if w not in stop_words and len(w) > 2
            ]

            if not keywords:
                return []

            search_query = ' '.join(keywords[:4])
            self.logger.info(f"Searching repo for: {search_query}")

            # GitHub code search
            try:
                code_results = self.github.search_code(
                    query=f"{search_query} repo:{repo.full_name}",
                )
                for item in list(code_results)[:5]:
                    try:
                        file_content = repo.get_contents(item.path)
                        decoded = file_content.decoded_content.decode(
                            'utf-8', errors='ignore'
                        )
                        targeted.append({
                            'name': item.name,
                            'path': item.path,
                            'content': decoded[:3000],
                        })
                        self.logger.info(f"Fetched targeted file: {item.path}")
                    except Exception as e:
                        self.logger.warning(f"Could not read {item.path}: {e}")
            except Exception as e:
                self.logger.warning(f"GitHub code search failed: {e}")

            # Fallback: walk repo tree for keyword matches
            if not targeted:
                self.logger.info("Code search empty — scanning repo tree")
                try:
                    tree = repo.get_git_tree(sha="HEAD", recursive=True).tree
                    for entry in tree:
                        if entry.type != 'blob':
                            continue
                        path_lower = entry.path.lower()
                        if any(kw in path_lower for kw in keywords):
                            try:
                                file_content = repo.get_contents(entry.path)
                                decoded = file_content.decoded_content.decode(
                                    'utf-8', errors='ignore'
                                )
                                targeted.append({
                                    'name': entry.path.split('/')[-1],
                                    'path': entry.path,
                                    'content': decoded[:3000],
                                })
                                self.logger.info(
                                    f"Fetched matching file: {entry.path}"
                                )
                                if len(targeted) >= 5:
                                    break
                            except Exception:
                                continue
                except Exception as e:
                    self.logger.warning(f"Tree scan failed: {e}")

        except Exception as e:
            self.logger.error(f"Targeted file fetch error: {e}")

        self.logger.info(f"Fetched {len(targeted)} targeted files")
        return targeted

    def _generate_targeted_answer(
        self,
        repo_name: str,
        targeted_files: List[Dict],
        query: str,
    ) -> str:
        """Generate a focused answer using targeted file contents."""
        files_text = ""
        for f in targeted_files:
            files_text += f"\n--- {f['path']} ---\n{f['content']}\n"

        prompt = f"""Analyze these specific files from the repository to answer
the user's question.

## Repository: {repo_name}

## Targeted File Contents
{files_text}

## User's Question
{query}

## Instructions
- Focus on answering the specific question using the file contents
- Be specific — reference actual code, functions, classes from the files
- Use markdown formatting with code blocks where relevant
"""

        response = self.agent(prompt)
        response_text = str(response)

        output = f"📦 **Codebase Deep Dive: {repo_name}**\n\n"
        output += f"🔍 *Targeted analysis — fetched {len(targeted_files)} "
        output += "specific files to answer your question*\n\n"
        output += response_text
        return output

    # ──────────────────────────────────────────────────────────
    # GitHub data extraction helpers
    # ──────────────────────────────────────────────────────────

    def _get_readme(self, repo) -> str:
        """Get repository README content"""
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode('utf-8')
            return content[:2000]
        except Exception:
            return "No README found"

    def _get_main_files(self, repo) -> List[str]:
        """Get main files from repository root"""
        try:
            return [content.name for content in repo.get_contents("")]
        except Exception:
            return []

    def _get_languages(self, repo) -> List[str]:
        """Get programming languages used in repository"""
        try:
            languages = repo.get_languages()
            return list(languages.keys())
        except Exception:
            return []

    def _get_dependencies(self, repo) -> Dict:
        """Get project dependencies"""
        try:
            dependencies = {}
            for file_name in ['requirements.txt', 'package.json', 'pom.xml', 'Gemfile', 'go.mod']:
                try:
                    content = repo.get_contents(file_name)
                    dep_content = content.decoded_content.decode('utf-8')
                    dependencies[file_name] = dep_content[:500]
                except Exception:
                    continue
            return dependencies
        except Exception:
            return {}

    def _format_response(self, response: str, repo_name: str) -> str:
        """Format response"""
        output = f"📦 **Codebase Onboarding: {repo_name}**\n\n"
        output += "*Fresh analysis from GitHub*\n\n"
        output += str(response)
        return output
