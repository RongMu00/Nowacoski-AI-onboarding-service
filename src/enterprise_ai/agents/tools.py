import logging
import os
import pickle

import dotenv
from google.oauth2.credentials import Credentials
from mcp.client.streamable_http import streamablehttp_client
from strands import tool

from enterprise_ai.agents.codebase_agent import CodebaseAgent
from enterprise_ai.agents.drive_agent import DriveAgent
from enterprise_ai.agents.tavily_agent import TavilyAgent

dotenv.load_dotenv()

@tool
def trigger_tavily_agent(context: str) -> str:
    """
    Triggers the Tavily Agent to perform a web search based on a natural language query.
    Returns a concise summary of the most relevant and reliable external information to support onboarding context, technical clarification, or open-ended questions.
    Use this when internal sources are insufficient or when real-time, up-to-date knowledge is required.

    Context must be a string. Also, run the tool at most 5 times. Make it concise and actionable.
    """
    agent = TavilyAgent()
    response = agent(context)
    return response

# @tool
# def trigger_google_drive_agent(drive_links: str):
#     """
#     Trigger the Google Drive agent.

#     Use this tool to create an onboarding plan based on documents in a Google Drive folder.
#     The `drive_links` parameter should be a string containing the Google Drive folder link.
#     """
#     print("Google Drive agent triggered")
#     # # Example usage
#     # credentials = Credentials.from_authorized_user_file('/Users/rongmu/Downloads/client_secret.json', [drive_links])

#     # # Initialize agent
#     # agent = DriveAgent(credentials)

#     # # Get onboarding plan from drive link
#     # drive_link = drive_links
#     # plan = agent.create_onboarding_plan(drive_link)

#     # # Print the steps
#     # return plan

#     try:
#         # 尝试加载凭证（如果存在）
#         creds = None
#         if os.path.exists('token.pickle'):
#             with open('token.pickle', 'rb') as token:
#                 creds = pickle.load(token)
#         #creds = credentials

#         # 初始化 DriveAgent（有无凭证都可以）
#         agent = DriveAgent(credentials=creds)

#         # 获取 onboarding 计划
#         plan = agent.create_onboarding_plan(drive_links)

#         # 格式化输出
#         result = "**Google Drive Folder Analysis**"

#         if plan.is_mock:
#             result += " (Demo Mode)\n\n"
#             result += "*Using simulated data - configure OAuth for real Drive access*\n\n"
#         else:
#             result += " (Live Data)\n\n"

#         result += "**Onboarding Plan:**\n\n"
#         for step in plan.steps:
#             result += f"{step}\n"

#         result += f"\n**Estimated Duration:** {plan.estimated_duration}"

#         if plan.error_message:
#             result += f"\n\nNote: {plan.error_message}"

#         return result

#     except Exception as e:
#         return f"Error: {str(e)}"

@tool
def trigger_codebase_agent(repo_url: str):
    """
    Trigger the Codebase agent.

    Use this tool to create an onboarding plan based on a GitHub repository.
    The `repo_url` parameter should be a string containing the GitHub repository URL.

    """
    try:
        # Example usage
        agent = CodebaseAgent(os.getenv('GITHUB_TOKEN'))

        # Analyze a GitHub repository
        onboarding = agent.create_onboarding_plan(repo_url)

        # 如果 result 是 AgentResult 对象，转换为字符串
        if hasattr(onboarding, '__str__'):
            return str(onboarding)
        else:
            return onboarding

    except Exception as e:
        return f"Error analyzing repository: {str(e)}"

    # Print the steps
    return onboarding

logger = logging.getLogger(__name__)

@tool
def trigger_google_drive_agent(drive_links: str) -> str:
    """
    Trigger the Google Drive agent to analyze onboarding documents.

    This function:
    1. Attempts to use service account (if configured)
    2. Falls back to user OAuth credentials (from token.pickle)
    3. Falls back to mock data if neither is available
    4. Creates an actionable onboarding plan from Google Drive documents

    Args:
        drive_links (str): Google Drive folder URL
                          Format: https://drive.google.com/drive/folders/FOLDER_ID

    Returns:
        str: Formatted onboarding plan or error message

    Example:
        >>> result = trigger_google_drive_agent(
        ...     'https://drive.google.com/drive/folders/1abc2def3ghi4jkl5mno'
        ... )
        >>> print(result)
        **Google Drive Folder Analysis**
        ...
    """
    logger.info(f"Google Drive agent triggered with link: {drive_links}")

    try:
        # Initialize credentials and agent
        creds = None
        service_account_path = '/Users/rongmu/Downloads/service_accounts_key.json'

        # Step 1: Try to load user OAuth credentials (token.pickle)
        if os.path.exists('token.pickle'):
            try:
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
                logger.info("✅ User OAuth credentials loaded from token.pickle")
            except Exception as e:
                logger.warning(f"⚠️  Could not load token.pickle: {e}")
                creds = None

        # Step 2: Initialize DriveAgent with fallback to service account
        try:
            agent = DriveAgent(
                credentials=creds,
                service_account_path=service_account_path
            )
            logger.info(f"✅ DriveAgent initialized (Auth method: {agent.auth_method})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DriveAgent: {e}")
            return _format_error_response(
                "Failed to initialize Drive Agent",
                str(e)
            )

        # Step 3: Validate the drive link
        if not drive_links or not _is_valid_drive_link(drive_links):
            logger.error(f"Invalid Drive link format: {drive_links}")
            return _format_error_response(
                "Invalid Google Drive Link",
                f"Expected format: https://drive.google.com/drive/folders/FOLDER_ID\nReceived: {drive_links}"
            )

        # Step 4: Get onboarding plan
        logger.info(f"Creating onboarding plan from: {drive_links}")
        plan = agent.create_onboarding_plan(drive_links)

        # Step 5: Format and return response
        return _format_plan_response(plan, agent.auth_method)

    except Exception as e:
        logger.exception(f"Unexpected error in trigger_google_drive_agent: {e}")
        return _format_error_response(
            "Unexpected Error",
            str(e)
        )


def _is_valid_drive_link(link: str) -> bool:
    """
    Validate if the provided link is a valid Google Drive folder link.

    Args:
        link (str): URL to validate

    Returns:
        bool: True if valid, False otherwise
    """
    valid_formats = [
        'drive.google.com/drive/folders/',
        'drive.google.com/open?id=',
    ]
    return any(fmt in link for fmt in valid_formats)


def _format_plan_response(plan, auth_method: str) -> str:
    """
    Format the onboarding plan into a user-friendly response.

    Args:
        plan: OnboardingPlan object from DriveAgent
        auth_method (str): Authentication method used ('service_account', 'user_oauth', or 'none')

    Returns:
        str: Formatted markdown response
    """
    result = "**📋 Google Drive Folder Analysis**\n\n"

    # Add authentication status
    if auth_method == 'service_account':
        result += "✅ *Using Service Account (Always available)*\n\n"
    elif auth_method == 'user_oauth':
        result += "✅ *Using User OAuth Credentials*\n\n"
    else:
        result += "ℹ️ *Using Demo Mode*\n\n"

    # Add accessibility status and warnings
    if not plan.is_accessible:
        result += "⚠️ **Access Issue:**\n"
        result += f"{plan.steps[0]}\n\n"
        if plan.error_message:
            result += f"**Error:** {plan.error_message}\n\n"
    else:
        # Add success indicator
        if plan.is_mock:
            result += "ℹ️ *Note: Using simulated onboarding plan. "
            result += "Authenticate with `authorize_drive.py` or configure service account for real data.*\n\n"

    # Add the actual plan
    result += "**📖 Onboarding Plan:**\n\n"

    if plan.steps:
        for i, step in enumerate(plan.steps, 1):
            # Format steps nicely
            if step.startswith(('📚', '🛠️', '📖', '👥', '💻', '🎯', '✅', '❌', '⚠️')):
                result += f"{step}\n"
            else:
                result += f"  {step}\n"
    else:
        result += "No steps available\n"

    # Add metadata
    result += f"\n**⏱️  Estimated Duration:** {plan.estimated_duration}\n"

    # Add additional info
    if plan.error_message and plan.is_accessible:
        result += f"\n**ℹ️  Note:** {plan.error_message}\n"

    return result


def _format_error_response(title: str, message: str) -> str:
    """
    Format an error response in user-friendly markdown.

    Args:
        title (str): Error title
        message (str): Error message/details

    Returns:
        str: Formatted error response
    """
    result = f"**❌ {title}**\n\n"
    result += f"**Error Details:**\n{message}\n\n"
    result += "**What to try:**\n"
    result += "1. Check that the Google Drive link is valid\n"
    result += "2. Make sure the folder is shared with your account\n"
    result += "3. Wait 5 minutes after sharing (permissions take time to sync)\n"
    result += "4. For API access, run: `python authorize_drive.py`\n"
    result += "5. Or set up a Service Account for production use\n"
    return result
