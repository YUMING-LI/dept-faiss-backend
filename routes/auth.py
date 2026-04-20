from flask import Blueprint, request, jsonify, make_response
from werkzeug.security import check_password_hash
import jwt
import os
import logging
import requests as _req
from datetime import datetime, timedelta, timezone

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

PROJECT = 'ihd-faiss'
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_EXPIRE_MINUTES', '60'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_EXPIRE_DAYS', '7'))
_COOKIE = 'ihd_refresh_token'

UPSTREAM_BASE_URL = os.getenv('UPSTREAM_BASE_URL', '').rstrip('/')
UPSTREAM_AUTH_MODE = os.getenv('UPSTREAM_AUTH_MODE', 'bearer').lower()

_VALID_PERMISSIONS = ('manager', 'editor', 'viewer')


def _secret():
    s = os.getenv('JWT_SECRET', '').strip()
    if not s:
        raise RuntimeError('JWT_SECRET env var is not set')
    return s


def _generate_tokens(account: str, permission: str = 'manager'):
    secret = _secret()
    now = datetime.now(timezone.utc)
    access = jwt.encode({
        'sub': account,
        'project': PROJECT,
        'permission': permission,
        'iat': now,
        'exp': now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }, secret, algorithm='HS256')
    refresh = jwt.encode({
        'sub': account,
        'project': PROJECT,
        'permission': permission,
        'type': 'refresh',
        'iat': now,
        'exp': now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    }, secret, algorithm='HS256')
    return access, refresh


def _set_refresh_cookie(resp, token: str):
    is_https = os.getenv('X_FORWARDED_PROTO', 'http') == 'https'
    resp.set_cookie(
        _COOKIE, token,
        httponly=True,
        secure=is_https,
        samesite='Lax',
        max_age=60 * 60 * 24 * REFRESH_TOKEN_EXPIRE_DAYS,
        path='/',
    )


def _upstream_headers(token: str) -> dict:
    if UPSTREAM_AUTH_MODE == 'x-access-token':
        return {'x-access-token': token}
    return {'Authorization': f'Bearer {token}'}


def _fetch_upstream_permission(upstream_token: str) -> str | None:
    """用 upstream token 查詢使用者的 ihd-faiss project_permission。"""
    for path in ('/api/auth/me', '/api/me', '/api/user/me'):
        try:
            r = _req.get(
                f'{UPSTREAM_BASE_URL}{path}',
                headers=_upstream_headers(upstream_token),
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()
                user = data.get('user') or data.get('data') or data
                pp = user.get('project_permission') or {}
                if isinstance(pp, dict) and 'ihd-faiss' in pp:
                    return pp['ihd-faiss']
        except Exception:
            pass
    return None


def _upstream_login(account: str, password: str) -> tuple[str | None, str | None]:
    """
    呼叫 upstream 登入 API。
    回傳 (upstream_token, ihd_faiss_permission)，失敗則回傳 (None, None)。
    """
    if not UPSTREAM_BASE_URL:
        return None, None

    try:
        resp = _req.post(
            f'{UPSTREAM_BASE_URL}/api/login',
            json={'account': account, 'password': password},
            headers={'Content-Type': 'application/json'},
            timeout=10,
        )
    except _req.exceptions.RequestException as e:
        logging.warning('Upstream unreachable: %s', e)
        return None, None

    if resp.status_code != 200:
        return None, None

    try:
        data = resp.json()
    except Exception:
        return None, None

    # 嘗試多種常見 token 欄位名稱
    upstream_token = (
        data.get('token') or
        data.get('access_token') or
        (data.get('data') or {}).get('token') or
        (data.get('data') or {}).get('access_token')
    )

    # 嘗試從登入回應直接取得 project_permission
    user_obj = data.get('user') or data.get('data') or {}
    pp = user_obj.get('project_permission') or {}

    if isinstance(pp, dict) and 'ihd-faiss' in pp:
        return upstream_token, pp['ihd-faiss']

    # 登入回應沒有 project_permission，用 token 額外查詢
    if upstream_token:
        perm = _fetch_upstream_permission(upstream_token)
        return upstream_token, perm

    return None, None


@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    account_input = (data.get('account') or '').strip()
    password_input = data.get('password') or ''

    if not account_input or not password_input:
        return jsonify({'success': False, 'message': '請輸入帳號與密碼'}), 400

    # ── 1. 嘗試 upstream 驗證（dept 使用者）───────────────────────
    if UPSTREAM_BASE_URL:
        _token, perm = _upstream_login(account_input, password_input)
        if perm and perm in _VALID_PERMISSIONS:
            access, refresh = _generate_tokens(account_input, permission=perm)
            resp = make_response(jsonify({
                'success': True,
                'access_token': access,
                'user': {'account': account_input, 'permission': perm},
            }), 200)
            _set_refresh_cookie(resp, refresh)
            return resp
        if _token is not None:
            # upstream 登入成功，但沒有 ihd-faiss 權限
            logging.warning("Upstream login ok for '%s' but no ihd-faiss permission", account_input)
            return jsonify({'success': False, 'message': '您的帳號沒有 ihd-faiss 存取權限'}), 403

    # ── 2. MANAGER_ACCOUNT fallback ─────────────────────────────
    manager_account = os.getenv('MANAGER_ACCOUNT', '').strip()
    manager_hash = os.getenv('MANAGER_PASSWORD_HASH', '').strip()

    if manager_account and manager_hash:
        if account_input == manager_account and check_password_hash(manager_hash, password_input):
            access, refresh = _generate_tokens(account_input, permission='manager')
            resp = make_response(jsonify({
                'success': True,
                'access_token': access,
                'user': {'account': account_input, 'permission': 'manager'},
            }), 200)
            _set_refresh_cookie(resp, refresh)
            return resp

    logging.warning("Login failed for account '%s' from %s", account_input, request.remote_addr)
    return jsonify({'success': False, 'message': '帳號或密碼錯誤'}), 401


@bp.route('/refresh', methods=['GET'])
def refresh():
    token = request.cookies.get(_COOKIE)
    if not token:
        return jsonify({'success': False, 'message': 'Refresh token missing'}), 401

    try:
        payload = jwt.decode(token, _secret(), algorithms=['HS256'])
        if payload.get('type') != 'refresh' or payload.get('project') != PROJECT:
            return jsonify({'success': False, 'message': '無效的 Refresh Token'}), 401

        account = payload['sub']
        permission = payload.get('permission', 'manager')
        access, new_refresh = _generate_tokens(account, permission=permission)
        resp = make_response(jsonify({
            'success': True,
            'access_token': access,
            'user': {'account': account, 'permission': permission},
        }), 200)
        _set_refresh_cookie(resp, new_refresh)
        return resp

    except jwt.ExpiredSignatureError:
        return jsonify({'success': False, 'message': 'Refresh token 已過期，請重新登入'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'success': False, 'message': '無效的 Refresh Token'}), 401
    except Exception as e:
        logging.error('Refresh error: %s', e)
        return jsonify({'success': False, 'message': '伺服器錯誤'}), 500


@bp.route('/logout', methods=['POST'])
def logout():
    resp = make_response(jsonify({'success': True, 'message': '登出成功'}), 200)
    resp.delete_cookie(_COOKIE, path='/', httponly=True, samesite='Lax')
    return resp
