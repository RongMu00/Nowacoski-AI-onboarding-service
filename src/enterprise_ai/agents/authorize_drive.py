import os
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authorize():
    """
    首次授权流程，生成 token.pickle

    FIX: This now uses the correct redirect_uri that matches your OAuth credentials
    """
    creds = None

    # 检查是否已有保存的令牌
    if os.path.exists('token.pickle'):
        print("发现已保存的令牌...")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # 如果没有有效令牌，进行授权
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("令牌已过期，正在刷新...")
            creds.refresh(Request())
        else:
            print("需要进行授权...")
            print("即将打开浏览器，请登录你的 Google 账号并授权")

            flow = InstalledAppFlow.from_client_secrets_file(
                '/Users/rongmu/Downloads/client_secret.json',  # Changed from hardcoded path
                SCOPES
            )

            # FIX: Set the redirect_uri to localhost with port 8080
            # This must match what's configured in Google Cloud Console
            creds = flow.run_local_server(port=8080, open_browser=True)

        # 保存令牌供下次使用
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
        print("✅ 令牌已保存到 token.pickle")

    print("✅ 授权成功！")
    return creds

if __name__ == '__main__':
    print("=" * 60)
    print("Google Drive 授权工具")
    print("=" * 60)
    print()

    # 检查 client_secret.json 是否存在
    if not os.path.exists('/Users/rongmu/Downloads/client_secret.json'):
        print("❌ 错误：找不到 client_secret.json")
        print()
        print("请从 Google Cloud Console 下载 OAuth 2.0 凭证：")
        print("1. https://console.cloud.google.com")
        print("2. APIs & Services → Credentials")
        print("3. 下载 OAuth 2.0 Client ID 的 JSON 文件")
        print("4. 重命名为 client_secret.json 并放在项目根目录")
        print()
        print("⚠️  IMPORTANT - Authorized redirect URIs setup:")
        print("   In Google Cloud Console, add these URIs to your OAuth 2.0 credentials:")
        print("   - http://localhost:8080/")
        print("   - http://localhost:8080")
        exit(1)

    authorize()

    print()
    print("=" * 60)
    print("现在可以使用 Google Drive 功能了！")
    print("=" * 60)
