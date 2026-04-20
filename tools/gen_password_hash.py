#!/usr/bin/env python3
"""
產生 MANAGER_PASSWORD_HASH 用的 werkzeug hash。

使用方式：
    python tools/gen_password_hash.py
    python tools/gen_password_hash.py mypassword

輸出結果填入 .env：
    MANAGER_PASSWORD_HASH=<hash>
"""
import sys
from werkzeug.security import generate_password_hash

if len(sys.argv) > 1:
    pw = sys.argv[1]
else:
    import getpass
    pw = getpass.getpass('請輸入密碼: ')

print(generate_password_hash(pw))
