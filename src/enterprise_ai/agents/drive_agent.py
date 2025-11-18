# import logging
# from typing import Dict, List
# from dataclasses import dataclass

# from strands import Agent
# from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError

# @dataclass
# class OnboardingPlan:
#     """Simple onboarding plan with ordered steps"""
#     steps: List[str]
#     estimated_duration: str

# class DriveAgent:
#     def __init__(self, credentials: Credentials):
#         """Initialize DriveAgent with Google Drive credentials"""
#         self.logger = logging.getLogger("drive_agent")
#         self.drive_service = build('drive', 'v3', credentials=credentials)
#         self.agent = Agent(system_prompt=self._get_system_prompt())

#         # self.logger = logging.getLogger("drive_agent")
#         # self.credentials = credentials
#         # self.is_authenticated = credentials is not None

#         # # 只在有凭证时才初始化 Drive service
#         # if self.is_authenticated:
#         #     try:
#         #         self.drive_service = build('drive', 'v3', credentials=credentials)
#         #         self.logger.info("Drive service initialized with credentials")
#         #     except Exception as e:
#         #         self.logger.warning(f"Failed to initialize Drive service: {e}")
#         #         self.is_authenticated = False
#         #         self.drive_service = None
#         # else:
#         #     self.drive_service = None
#         #     self.logger.info("Drive agent initialized in mock mode (no credentials)")

#         # self.agent = Agent(system_prompt=self._get_system_prompt())

#     def _get_system_prompt(self) -> str:
#         return """You are an onboarding assistant. Given content from various documents:
#         1. Analyze and understand the onboarding requirements
#         2. Create a clear, ordered list of onboarding steps
#         3. Ensure steps are concrete and actionable
#         4. Include estimated time for each step
#         Format your response as a numbered list of steps."""

#     def create_onboarding_plan(self, drive_link: str) -> OnboardingPlan:
#         """Create onboarding plan from Google Drive folder contents"""
#         # 如果没有认证，使用模拟模式
#         if not self.is_authenticated:
#             self.logger.info("Using mock mode for Drive analysis")
#             return self._create_mock_onboarding_plan(drive_link)

#         try:
#             # Extract folder ID and get contents
#             folder_id = drive_link.split('/folders/')[-1].split('?')[0]
#             documents = self._get_folder_contents(folder_id)

#             # Collect all content
#             all_content = []
#             for doc in documents:
#                 content = self._get_document_content(doc['id'])
#                 all_content.append(content)

#             # Generate plan using combined content
#             prompt = f"Create an onboarding plan based on these documents:\n{''.join(all_content)}"
#             response = self.agent(prompt)

#             # Parse steps from response
#             steps = [step.strip() for step in response.content.split('\n') if step.strip()]
#             return OnboardingPlan(
#                 steps=steps,
#                 estimated_duration="1-2 weeks"  # You can make this more dynamic
#             )

#         except Exception as e:
#             self.logger.error(f"Error processing drive folder: {e}")
#             self.logger.info("Falling back to mock mode")
#             # 发生错误时，降级到模拟模式
#             return self._create_mock_onboarding_plan(drive_link)
#             #raise

#     def _create_mock_onboarding_plan(self, drive_link: str) -> OnboardingPlan:
#         """
#         Create a mock onboarding plan based on best practices

#         This is used when Drive API is not available.
#         """
#         mock_steps = [
#             "📚 Week 1: Company & Culture Orientation",
#             "  - Review company mission, vision, and values",
#             "  - Understand organizational structure and team dynamics",
#             "  - Complete HR onboarding and paperwork",
#             "  - Set up communication tools (Slack, Email, etc.)",
#             "",
#             "🛠️ Week 1-2: Technical Environment Setup",
#             "  - Install required development tools and IDE",
#             "  - Set up version control (Git) and access repositories",
#             "  - Configure local development environment",
#             "  - Obtain necessary access permissions and credentials",
#             "",
#             "📖 Week 2-3: Codebase Familiarization",
#             "  - Review architecture and system design documentation",
#             "  - Understand code structure and organization",
#             "  - Study coding standards and best practices",
#             "  - Learn testing and deployment processes",
#             "",
#             "👥 Week 2-3: Team Integration",
#             "  - Meet with team members and key stakeholders",
#             "  - Shadow experienced developers",
#             "  - Participate in team meetings and stand-ups",
#             "  - Connect with assigned mentor or buddy",
#             "",
#             "💻 Week 3-4: Hands-On Work",
#             "  - Start with small, well-defined tasks (bug fixes, documentation)",
#             "  - Submit first pull request and participate in code review",
#             "  - Work on increasingly complex features",
#             "  - Contribute to team discussions and planning",
#             "",
#             "🎯 Week 4+: Continuous Learning",
#             "  - Take on ownership of feature development",
#             "  - Participate in design discussions",
#             "  - Share knowledge with team members",
#             "  - Set personal development goals with mentor"
#         ]

#         return OnboardingPlan(
#             steps=mock_steps,
#             estimated_duration="4+ weeks for full onboarding",
#             is_mock=True
#         )

#     def _get_folder_contents(self, folder_id: str) -> List[Dict]:
#         """Get list of files in a Google Drive folder"""
#         if not self.drive_service:
#             return []

#         try:
#             results = self.drive_service.files().list(
#                 q=f"'{folder_id}' in parents",
#                 fields="files(id, name, mimeType)"
#             ).execute()
#             return results.get('files', [])
#         except Exception as e:
#             self.logger.error(f"Error getting folder contents: {e}")
#             return []

#     def _get_document_content(self, file_id: str) -> str:
#         """Get content from a Google Drive document"""
#         if not self.drive_service:
#             return ""

#         try:
#             # This is a simplified version - you'd need to handle different file types
#             request = self.drive_service.files().get_media(fileId=file_id)
#             content = request.execute()
#             return content.decode('utf-8') if isinstance(content, bytes) else str(content)
#         except Exception as e:
#             self.logger.error(f"Error getting document content: {e}")
#             return ""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from strands import Agent


@dataclass
class OnboardingPlan:
    """Simple onboarding plan with ordered steps"""
    steps: List[str]
    estimated_duration: str
    is_accessible: bool = True
    is_mock: bool = False
    error_message: Optional[str] = None

class DriveAgent:
    def __init__(self, credentials: Credentials = None, service_account_path: str = None):
        """
        Initialize DriveAgent with optional credentials

        Args:
            credentials: Google OAuth credentials (from user authorization)
            service_account_path: Path to service account JSON file (optional)
        """
        self.logger = logging.getLogger("drive_agent")

        # Try to use provided credentials
        if credentials:
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.is_authenticated = True
            self.auth_method = "user_oauth"
            self.logger.info("Using user OAuth credentials")

        # Try to load service account as fallback
        elif service_account_path:
            try:
                import os

                from google.oauth2 import service_account

                if os.path.exists(service_account_path):
                    sa_credentials = service_account.Credentials.from_service_account_file(
                        service_account_path,
                        scopes=['https://www.googleapis.com/auth/drive.readonly']
                    )
                    self.drive_service = build('drive', 'v3', credentials=sa_credentials)
                    self.is_authenticated = True
                    self.auth_method = "service_account"
                    self.logger.info("Using service account credentials")
                else:
                    self.drive_service = None
                    self.is_authenticated = False
                    self.auth_method = "none"
                    self.logger.info("Service account file not found, running in mock mode")
            except Exception as e:
                self.logger.warning(f"Failed to load service account: {e}")
                self.drive_service = None
                self.is_authenticated = False
                self.auth_method = "none"

        # No credentials available
        else:
            self.drive_service = None
            self.is_authenticated = False
            self.auth_method = "none"
            self.logger.info("No credentials provided, running in mock mode")

        self.agent = Agent(system_prompt=self._get_system_prompt())

    def _get_system_prompt(self) -> str:
        return """You are an onboarding assistant. Given content from various documents:
        1. Analyze and understand the onboarding requirements
        2. Create a clear, ordered list of onboarding steps
        3. Ensure steps are concrete and actionable
        4. Include estimated time for each step
        Format your response as a numbered list of steps."""

    def create_onboarding_plan(self, drive_link: str) -> OnboardingPlan:
        """
        Create onboarding plan from Google Drive folder contents

        Args:
            drive_link: Google Drive folder link (e.g., https://drive.google.com/drive/folders/FOLDER_ID)

        Returns:
            OnboardingPlan object with steps and metadata
        """

        # If not authenticated, use mock mode
        if not self.is_authenticated:
            self.logger.info("Not authenticated - using mock mode")
            return self._create_mock_onboarding_plan(drive_link)

        try:
            # Extract folder ID from link
            folder_id = self._extract_folder_id(drive_link)
            if not folder_id:
                raise ValueError("Could not extract folder ID from link")

            self.logger.info(f"Attempting to access folder: {folder_id}")

            # Get folder contents
            documents = self._get_folder_contents(folder_id)

            if not documents:
                return OnboardingPlan(
                    steps=[
                        "⚠️ Folder is empty or not accessible",
                        f"Auth method: {self.auth_method}",
                        "",
                        "Possible reasons:",
                        "• The folder contains no files",
                        "• You don't have permission to access this folder",
                        "• The folder link is invalid"
                    ],
                    estimated_duration="N/A",
                    is_accessible=False,
                    error_message="Empty or inaccessible folder"
                )

            # Collect content from documents
            all_content = []
            for doc in documents:
                try:
                    content = self._get_document_content(doc['id'], doc.get('mimeType'))
                    if content:
                        all_content.append(f"[{doc['name']}]\n{content}\n\n")
                except Exception as e:
                    self.logger.warning(f"Could not read document {doc['name']}: {e}")
                    continue

            if not all_content:
                return OnboardingPlan(
                    steps=[
                        "⚠️ Could not read any documents from folder",
                        "The folder may contain files we cannot process (e.g., images, videos)"
                    ],
                    estimated_duration="N/A",
                    is_accessible=False,
                    error_message="No readable documents found"
                )

            # Generate onboarding plan
            prompt = f"Create an onboarding plan based on these documents:\n\n{''.join(all_content)}"
            response = self.agent(prompt)

            # Parse response
            steps = [step.strip() for step in str(response).split('\n') if step.strip()]

            return OnboardingPlan(
                steps=steps if steps else ["No plan generated"],
                estimated_duration="1-2 weeks",
                is_accessible=True,
                is_mock=False
            )

        except HttpError as e:
            error_code = e.resp.status

            if error_code == 403:
                return OnboardingPlan(
                    steps=[
                        "❌ Access Denied (403)",
                        "",
                        "You don't have permission to access this folder.",
                        "",
                        "Solutions:",
                        "1. Ask the folder owner to share it with you",
                        f"   Your account: (check your Google account)",
                        "",
                        "2. If using service account authentication:",
                        "   Ask the folder owner to share it with the service account email",
                    ],
                    estimated_duration="N/A",
                    is_accessible=False,
                    error_message=f"Permission denied (403)"
                )

            elif error_code == 404:
                return OnboardingPlan(
                    steps=[
                        "❌ Folder Not Found (404)",
                        "",
                        "The folder doesn't exist or has been deleted.",
                        "Please check the folder link and try again."
                    ],
                    estimated_duration="N/A",
                    is_accessible=False,
                    error_message=f"Folder not found (404)"
                )

            else:
                return OnboardingPlan(
                    steps=[
                        f"❌ API Error ({error_code})",
                        f"Message: {str(e)}",
                        "",
                        "Please check the folder link and try again."
                    ],
                    estimated_duration="N/A",
                    is_accessible=False,
                    error_message=f"API error: {error_code}"
                )

        except Exception as e:
            self.logger.error(f"Error processing drive folder: {e}")
            self.logger.info("Falling back to mock mode")
            return self._create_mock_onboarding_plan(drive_link)

    def _extract_folder_id(self, drive_link: str) -> Optional[str]:
        """Extract folder ID from Google Drive link"""
        try:
            # Handle different URL formats
            if '/folders/' in drive_link:
                return drive_link.split('/folders/')[-1].split('?')[0].split('#')[0]
            elif '/drive/folders/' in drive_link:
                return drive_link.split('/drive/folders/')[-1].split('?')[0].split('#')[0]
            else:
                return None
        except Exception as e:
            self.logger.error(f"Could not extract folder ID: {e}")
            return None

    def _create_mock_onboarding_plan(self, drive_link: str) -> OnboardingPlan:
        """
        Create a mock onboarding plan based on best practices

        This is used when Drive API is not available or when accessing real data fails.
        """
        mock_steps = [
            "📚 Week 1: Company & Culture Orientation",
            "  - Review company mission, vision, and values",
            "  - Understand organizational structure and team dynamics",
            "  - Complete HR onboarding and paperwork",
            "  - Set up communication tools (Slack, Email, etc.)",
            "",
            "🛠️ Week 1-2: Technical Environment Setup",
            "  - Install required development tools and IDE",
            "  - Set up version control (Git) and access repositories",
            "  - Configure local development environment",
            "  - Obtain necessary access permissions and credentials",
            "",
            "📖 Week 2-3: Codebase Familiarization",
            "  - Review architecture and system design documentation",
            "  - Understand code structure and organization",
            "  - Study coding standards and best practices",
            "  - Learn testing and deployment processes",
            "",
            "👥 Week 2-3: Team Integration",
            "  - Meet with team members and key stakeholders",
            "  - Shadow experienced developers",
            "  - Participate in team meetings and stand-ups",
            "  - Connect with assigned mentor or buddy",
            "",
            "💻 Week 3-4: Hands-On Work",
            "  - Start with small, well-defined tasks (bug fixes, documentation)",
            "  - Submit first pull request and participate in code review",
            "  - Work on increasingly complex features",
            "  - Contribute to team discussions and planning",
            "",
            "🎯 Week 4+: Continuous Learning",
            "  - Take on ownership of feature development",
            "  - Participate in design discussions",
            "  - Share knowledge with team members",
            "  - Set personal development goals with mentor"
        ]

        return OnboardingPlan(
            steps=mock_steps,
            estimated_duration="4+ weeks for full onboarding",
            is_accessible=False,
            is_mock=True,
            error_message="Using simulated onboarding plan (real data not available)"
        )

    def _get_folder_contents(self, folder_id: str) -> List[Dict]:
        """Get list of files in a Google Drive folder"""
        if not self.drive_service:
            return []

        try:
            results = self.drive_service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name, mimeType)",
                pageSize=100
            ).execute()

            files = results.get('files', [])
            self.logger.info(f"Found {len(files)} files in folder")
            return files

        except HttpError as e:
            self.logger.error(f"Error getting folder contents: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error getting folder contents: {e}")
            return []

    def _get_document_content(self, file_id: str, mime_type: str = None) -> str:
        """
        Get content from a Google Drive document/file

        Supports:
        - Google Docs
        - Google Sheets
        - Text files
        - PDFs (limited)
        """
        if not self.drive_service:
            return ""

        try:
            # Handle Google Docs
            if mime_type == 'application/vnd.google-apps.document':
                request = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
                content = request.execute()
                return content.decode('utf-8') if isinstance(content, bytes) else str(content)

            # Handle Google Sheets
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                request = self.drive_service.files().export_media(
                    fileId=file_id,
                    mimeType='text/csv'
                )
                content = request.execute()
                return content.decode('utf-8') if isinstance(content, bytes) else str(content)

            # Handle other file types
            else:
                request = self.drive_service.files().get_media(fileId=file_id)
                content = request.execute()
                return content.decode('utf-8') if isinstance(content, bytes) else str(content)

        except HttpError as e:
            if e.resp.status == 403:
                self.logger.warning(f"Cannot read file {file_id}: Access denied")
            else:
                self.logger.warning(f"Cannot read file {file_id}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Error reading file {file_id}: {e}")
            return ""
