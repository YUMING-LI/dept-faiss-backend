from __future__ import annotations

import os
import logging
import functools
import requests as _req
from flask import Blueprint, request, jsonify
import jwt

bp = Blueprint('perm', __name__, url_prefix='/api/admin')

UPSTREAM_BASE_URL = os.getenv('UPSTREAM_BASE_URL', '').rstrip('/')
UPSTREAM_PROJECT  = os.getenv('UPSTREAM_PROJECT', 'ihd-dept')


def _require_manager(f):
    """Decorator: 要求 ihd-faiss manager JWT"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'success': False, 'message': '未授權'}), 401
        token = auth[7:]
        secret = os.getenv('JWT_SECRET', '').strip()
        try:
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            if payload.get('project') != 'ihd-faiss' or payload.get('permission') != 'manager':
                return jsonify({'success': False, 'message': '需要 manager 權限'}), 403
        except Exception:
            return jsonify({'success': False, 'message': '無效的 Token'}), 401
        return f(*args, **kwargs)
    return decorated


def _get_admin_token() -> str | None:
    """用設定的服務帳號登入 dept 系統，取得 access_token。"""
    account  = os.getenv('UPSTREAM_ADMIN_ACCOUNT', '').strip()
    password = os.getenv('UPSTREAM_ADMIN_PASSWORD', '').strip()
    if not UPSTREAM_BASE_URL or not account or not password:
        return None
    try:
        resp = _req.post(
            f'{UPSTREAM_BASE_URL}/api/{UPSTREAM_PROJECT}/login?project={UPSTREAM_PROJECT}',
            json={'account': account, 'password': password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get('access_token')
        logging.warning('Upstream admin login failed: %s', resp.status_code)
    except Exception as e:
        logging.warning('Upstream admin login error: %s', e)
    return None


def _admin_headers(token: str) -> dict:
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}


# ──────────────────────────────────────────────────────────────
# GET /api/admin/dept-users
# 列出擁有 ihd-faiss 權限的 dept 使用者
# ──────────────────────────────────────────────────────────────
@bp.route('/dept-users', methods=['GET'])
@_require_manager
def list_dept_users():
    token = _get_admin_token()
    if not token:
        return jsonify({'success': False,
                        'message': 'UPSTREAM_ADMIN_ACCOUNT / UPSTREAM_ADMIN_PASSWORD 未設定'}), 503

    try:
        resp = _req.post(
            f'{UPSTREAM_BASE_URL}/api/{UPSTREAM_PROJECT}/users',
            headers=_admin_headers(token),
            json={'project': 'ihd-faiss'},
            timeout=10,
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'連線失敗: {e}'}), 503

    if resp.status_code != 200:
        return jsonify({'success': False, 'message': f'查詢失敗 ({resp.status_code})'}), 502

    raw = resp.json()
    result = raw.get('result') or {}
    rows   = result.get('rows', []) if isinstance(result, dict) else []

    users = [
        {
            'id':         u.get('id'),
            'account':    u.get('account', ''),
            'name':       u.get('name', ''),
            'permission': (u.get('project_permission') or {}).get('ihd-faiss', ''),
        }
        for u in rows
    ]
    return jsonify({'success': True, 'users': users}), 200


# ──────────────────────────────────────────────────────────────
# PUT /api/admin/dept-users/permission
# 設定（或撤銷）指定 dept 使用者的 ihd-faiss 權限
# Body: { account: "...", permission: "manager"/"editor"/"viewer"/"" }
# permission="" 代表撤銷
# ──────────────────────────────────────────────────────────────
@bp.route('/dept-users/permission', methods=['PUT'])
@_require_manager
def set_user_permission():
    body       = request.get_json(silent=True) or {}
    account    = (body.get('account') or '').strip()
    permission = (body.get('permission') or '').strip()

    if not account:
        return jsonify({'success': False, 'message': '缺少 account'}), 400
    if permission and permission not in ('manager', 'editor', 'viewer'):
        return jsonify({'success': False, 'message': '無效的 permission，可用值：manager / editor / viewer / (空字串=撤銷)'}), 400

    token = _get_admin_token()
    if not token:
        return jsonify({'success': False,
                        'message': 'UPSTREAM_ADMIN_ACCOUNT / UPSTREAM_ADMIN_PASSWORD 未設定'}), 503

    # 1. 以帳號查找使用者，取得 id 與目前的 project_permission
    try:
        r_find = _req.post(
            f'{UPSTREAM_BASE_URL}/api/{UPSTREAM_PROJECT}/users',
            headers=_admin_headers(token),
            json={'account': account},
            timeout=10,
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'查詢使用者失敗: {e}'}), 503

    if r_find.status_code != 200:
        return jsonify({'success': False, 'message': f'找不到使用者 ({r_find.status_code})'}), 404

    # POST /users 回傳 result 可能是 dict（單筆）或 {rows: [...]}（多筆）
    raw_result = r_find.json().get('result') or {}
    if isinstance(raw_result, dict) and 'id' in raw_result:
        user = raw_result
    elif isinstance(raw_result, dict) and 'rows' in raw_result:
        rows = raw_result.get('rows') or []
        user = rows[0] if rows else None
    else:
        user = None

    if not user or not user.get('id'):
        return jsonify({'success': False, 'message': f'找不到帳號：{account}'}), 404

    user_id    = user['id']
    current_pp = dict(user.get('project_permission') or {})

    # 2. 合併 ihd-faiss 權限
    if permission:
        current_pp['ihd-faiss'] = permission
    else:
        current_pp.pop('ihd-faiss', None)

    # 3. 只送 project_permission，不帶 campus/department，避免 dept 端 marshmallow null 驗證問題
    try:
        r_put = _req.put(
            f'{UPSTREAM_BASE_URL}/api/{UPSTREAM_PROJECT}/users/{user_id}',
            headers=_admin_headers(token),
            json={'project_permission': current_pp},
            timeout=10,
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'更新失敗: {e}'}), 503

    if r_put.status_code != 200:
        return jsonify({'success': False, 'message': f'更新失敗: {r_put.text}'}), 502

    action = f'設定為 {permission}' if permission else '已撤銷'
    return jsonify({'success': True, 'message': f'{account} 的 ihd-faiss 權限{action}'}), 200
