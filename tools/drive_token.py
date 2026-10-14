"""Print a fresh Google access token from the colab CLI's own OAuth login.

Must run under the CLI's interpreter (it imports colab_cli):
    env -u PYTHONPATH ~/.local/share/uv/tools/google-colab-cli/bin/python3 tools/drive_token.py

The token carries the CLI's scopes, including drive.file: Drive files *this
OAuth client created* are readable/writable, nothing else. Valid ~1 hour.
Prints only the token, to stdout; callers must not log it.
"""
from colab_cli.auth import get_credentials

session = get_credentials()  # requests.AuthorizedSession; refreshes as needed
creds = session.credentials
if not creds.valid:
    import google.auth.transport.requests

    creds.refresh(google.auth.transport.requests.Request())
print(creds.token)
