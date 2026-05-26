import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from github import Github
from strands import Agent

# Import VectorDB singleton for caching
from enterprise_ai.storage.vector_store import get_vector_store

logger = logging.getLogger("codebase_agent")


@dataclass
class CodebaseOnboarding:
    """Represents an onboarding plan for a codebase"""
    repository_name: str
    main_technologies: List[str]
    setup_steps: List[str]
    key_components: List[Dict[str, str]]
    estimated_study_time: str
    cached: bool = False
    source: str = "fresh"


class CodebaseAgent:
    def __init__(self, github_token: str):
        """Initialize CodebaseAgent with GitHub token and VectorDB"""
        self.logger = logging.getLogger("codebase_agent")
        self.github = Github(github_token)
        self.agent = Agent(system_prompt=self._get_system_prompt())

        # Initialize VectorDB singleton for caching
        try:
            self.vector_store = get_vector_store()
            self.logger.info("Vector store initialized for codebase caching")
        except Exception as e:
            self.logger.warning(f"Vector store not available: {e}")
            self.vector_store = None

    def _get_system_prompt(self) -> str:
        return """You are a codebase analysis assistant. For a given GitHub repository:
        1. Analyze the main technologies and frameworks used
        2. Create clear setup instructions for developers
        3. Identify key components and their purposes
        4. Suggest a learning path for new developers
        5. Estimate time needed to understand the codebase

        Format your response as a structured list with clear sections."""

    def create_onboarding_plan(self, repo_url: str, query: str = None) -> str:
        """Create onboarding plan from GitHub repository with VectorDB caching.

        Args:
            repo_url: GitHub repository URL
            query: Optional specific question about the repo. When provided
                   and a cached analysis exists, the agent checks whether the
                   cache covers the question. If not, it fetches targeted files
                   from GitHub to supplement the cached analysis (hybrid mode).
        """
        try:
            # Extract repo information from URL
            repo_name = repo_url.split('github.com/')[-1].strip('/')
            repo_id = repo_name.lower().replace('/', '-')

            self.logger.info(f"📦 Processing repository: {repo_name}")

            # STEP 1: Check VectorDB cache first
            cached_content = self._get_cached_repo_content(repo_id)

            if cached_content:
                # If no specific query, return cached analysis (fast path)
                if not query:
                    self.logger.info("Using cached analysis from VectorDB")
                    return self._format_cached_response(cached_content, repo_name, True)

                # Hybrid mode: check if cache covers the specific question
                relevance = self._check_cache_relevance(cached_content, query)
                self.logger.info(
                    f"Cache relevance to query: {relevance:.0%} (threshold: 50%)"
                )

                if relevance >= 0.5:
                    self.logger.info("Cached analysis covers the question — using cache")
                    return self._format_cached_response(cached_content, repo_name, True)

                # Cache doesn't cover this question — fetch targeted files
                self.logger.info(
                    "Cached analysis insufficient for this question — "
                    "fetching targeted files from GitHub"
                )
                repo = self.github.get_repo(repo_name)
                targeted_content = self._fetch_targeted_files(repo, query)

                if targeted_content:
                    # Generate focused answer combining cache + targeted files
                    return self._generate_targeted_answer(
                        repo_name, repo_url, repo_id,
                        cached_content, targeted_content, query
                    )
                else:
                    # Couldn't find relevant files — fall back to cached analysis
                    self.logger.info("No targeted files found — using cached analysis")
                    return self._format_cached_response(cached_content, repo_name, True)

            # STEP 2: Fetch from GitHub (no cache exists at all)
            self.logger.info("Fetching repository data from GitHub...")
            repo = self.github.get_repo(repo_name)

            # Collect repository information
            readme = self._get_readme(repo)
            main_files = self._get_main_files(repo)
            dependencies = self._get_dependencies(repo)
            languages = self._get_languages(repo)

            # STEP 3: Generate analysis
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
            response_text = str(response)

            # STEP 4: Cache the generic analysis
            if self.vector_store:
                doc_id = self.vector_store.store_document(
                    content=response_text,
                    source=f"Repository: {repo_name}",
                    folder_id=f"github-{repo_id}",
                    metadata={
                        "type": "codebase_analysis",
                        "repo_url": repo_url,
                        "repo_name": repo_name,
                        "languages": languages,
                        "description": repo.description
                    }
                )

                if doc_id:
                    self.logger.info(f"Cached codebase analysis: {doc_id}")

            # STEP 5: If user asked a specific question, also fetch targeted files
            if query:
                self.logger.info(
                    "First fetch + specific query — fetching targeted files too"
                )
                targeted_content = self._fetch_targeted_files(repo, query)
                if targeted_content:
                    return self._generate_targeted_answer(
                        repo_name, repo_url, repo_id,
                        response_text, targeted_content, query
                    )

            return self._format_response(response_text, repo_name, False)

        except Exception as e:
            self.logger.error(f"Error analyzing repository: {e}")
            return f"Error analyzing repository: {str(e)}"

    def _get_cached_repo_content(self, repo_id: str) -> Optional[str]:
        """Retrieve cached repository analysis from VectorDB"""
        if not self.vector_store or self.vector_store.collection is None:
            return None

        try:
            folder_id = f"github-{repo_id}"
            docs = self.vector_store.get_documents_by_folder(folder_id, limit=1)

            if docs:
                self.logger.info(f"Found cached repository analysis")
                return docs[0]['content']

            return None

        except Exception as e:
            self.logger.warning(f"Could not retrieve cached repo: {e}")
            return None

    def _check_cache_relevance(self, cached_content: str, query: str) -> float:
        """Check how well the cached analysis covers the specific question.

        Uses VectorDB semantic similarity between the query and cached content.
        Returns a similarity score between 0.0 and 1.0.
        """
        if not self.vector_store:
            return 0.0

        try:
            results = self.vector_store.search_similar(
                query=query, top_k=1, threshold=0.0
            )
            if results:
                return results[0].get('similarity', 0.0)
            return 0.0
        except Exception as e:
            self.logger.warning(f"Relevance check failed: {e}")
            return 0.0

    def _fetch_targeted_files(self, repo, query: str) -> List[Dict]:
        """Fetch specific files from GitHub that are relevant to the query.

        Uses GitHub code search to find files matching query keywords,
        then fetches their content. Returns list of {name, path, content}.
        """
        targeted = []

        try:
            # Extract keywords from query for file search
            stop_words = {
                'what', 'does', 'the', 'how', 'is', 'in', 'a', 'an', 'for',
                'to', 'of', 'this', 'that', 'do', 'can', 'you', 'explain',
                'about', 'tell', 'me', 'file', 'code', 'function', 'class',
            }
            keywords = [
                w for w in query.lower().split()
                if w not in stop_words and len(w) > 2
            ]

            if not keywords:
                return []

            search_query = ' '.join(keywords[:4])  # Limit to top 4 keywords
            self.logger.info(f"Searching repo for: {search_query}")

            # Search for files using GitHub code search
            try:
                code_results = self.github.search_code(
                    query=f"{search_query} repo:{repo.full_name}",
                )
                for item in list(code_results)[:5]:  # Top 5 matching files
                    try:
                        file_content = repo.get_contents(item.path)
                        decoded = file_content.decoded_content.decode(
                            'utf-8', errors='ignore'
                        )
                        # Limit per file to avoid token explosion
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

            # Also try to find files by walking directories for keyword matches
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
        repo_url: str,
        repo_id: str,
        cached_analysis: str,
        targeted_files: List[Dict],
        query: str,
    ) -> str:
        """Generate a focused answer combining cached analysis + targeted files.

        Uses the LLM to synthesize the high-level cached analysis with
        specific file contents to answer the user's detailed question.
        """
        # Format targeted file content
        files_text = ""
        for f in targeted_files:
            files_text += f"\n--- {f['path']} ---\n{f['content']}\n"

        prompt = f"""You have a cached high-level analysis of a repository AND
specific file contents fetched because the cached analysis didn't fully cover
the user's question. Synthesize both to provide a detailed, accurate answer.

## Cached Repository Analysis (high-level)
{cached_analysis[:2000]}

## Targeted File Contents (fetched for this specific question)
{files_text}

## User's Question
{query}

## Instructions
- Focus on answering the specific question using the targeted file contents
- Use the cached analysis for overall context
- Be specific — reference actual code, functions, classes from the files
- Use markdown formatting with code blocks where relevant
"""

        response = self.agent(prompt)
        response_text = str(response)

        # Cache this targeted analysis too
        if self.vector_store:
            doc_id = self.vector_store.store_document(
                content=response_text,
                source=f"Targeted: {repo_name} — {query[:60]}",
                folder_id=f"github-{repo_id}",
                metadata={
                    "type": "codebase_targeted_analysis",
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "query": query,
                    "files_analyzed": [f['path'] for f in targeted_files],
                },
            )
            if doc_id:
                self.logger.info(f"Cached targeted analysis: {doc_id}")

        output = f"📦 **Codebase Deep Dive: {repo_name}**\n\n"
        output += f"🔍 *Targeted analysis — fetched {len(targeted_files)} "
        output += "specific files to answer your question*\n\n"
        output += response_text
        return output

    def _get_readme(self, repo) -> str:
        """Get repository README content"""
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode('utf-8')
            return content[:2000]  # Limit to first 2000 chars
        except:
            return "No README found"

    def _get_main_files(self, repo) -> List[str]:
        """Get main files from repository root"""
        try:
            return [content.name for content in repo.get_contents("")]
        except:
            return []

    def _get_languages(self, repo) -> List[str]:
        """Get programming languages used in repository"""
        try:
            languages = repo.get_languages()
            return list(languages.keys())
        except:
            return []

    def _get_dependencies(self, repo) -> Dict:
        """Get project dependencies"""
        try:
            dependencies = {}

            # Common dependency files
            for file_name in ['requirements.txt', 'package.json', 'pom.xml', 'Gemfile', 'go.mod']:
                try:
                    content = repo.get_contents(file_name)
                    dep_content = content.decoded_content.decode('utf-8')
                    dependencies[file_name] = dep_content[:500]  # First 500 chars
                except:
                    continue

            return dependencies
        except:
            return {}

    def search_codebase(self, query: str) -> str:
        """Search cached codebase analyses semantically"""
        if not self.vector_store:
            return "Vector store not available"

        try:
            results = self.vector_store.search_similar(
                query=query,
                top_k=3,
                threshold=0.3
            )

            if not results:
                return f"No codebase analyses found matching: {query}"

            output = f"**Found {len(results)} related repositories:**\n\n"

            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                source = result.get('source', 'Unknown')
                metadata = result.get('metadata', {})

                output += f"{i}. **{source}** ({similarity:.0%} relevant)\n"
                if metadata.get('repo_url'):
                    output += f"   URL: {metadata['repo_url']}\n"
                output += f"   Languages: {', '.join(metadata.get('languages', []))}\n\n"

            return output

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return f"Search error: {e}"

    def _format_response(self, response: str, repo_name: str, from_cache: bool) -> str:
        """Format response"""
        output = f"📦 **Codebase Onboarding: {repo_name}**\n\n"

        if from_cache:
            output += "*Using cached analysis (for fresh data, clear cache)*\n\n"
        else:
            output += "*Fresh analysis from GitHub*\n\n"

        output += str(response)
        return output

    def _format_cached_response(self, content: str, repo_name: str, from_cache: bool) -> str:
        """Format cached response"""
        output = f"📦 **Codebase Onboarding: {repo_name}**\n\n"
        output += "📦 *Using cached analysis from previous fetch*\n\n"
        output += content
        return output

    def clear_cache(self, repo_id: str) -> int:
        """Clear cache for a specific repository"""
        if self.vector_store:
            folder_id = f"github-{repo_id.lower().replace('/', '-')}"
            deleted = self.vector_store.delete_folder_documents(folder_id)
            self.logger.info(f"Cleared {deleted} cached repository analyses")
            return deleted
        return 0
