from flask import Blueprint, request, jsonify, make_response
from werkzeug.security import check_password_hash
import jwt
import os
import logging
from datetime import datetime, timedelta, timezone

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

PROJECT = 'ihd-faiss'
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_EXPIRE_MINUTES', '60'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('JWT_REFRESH_EXPIRE_DAYS', '7'))
_COOKIE = 'ihd_refresh_token'


def _secret():
    s = os.getenv('JWT_SECRET', '').strip()
    if not s:
        raise RuntimeError('JWT_SECRET env var is not set')
    return s


def _generate_tokens(account: str):
    secret = _secret()
    now = datetime.now(timezone.utc)
    access = jwt.encode({
        'sub': account,
        'project': PROJECT,
        'permission': 'manager',
        'iat': now,
        'exp': now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }, secret, algorithm='HS256')
    refresh = jwt.encode({
        'sub': account,
        'project': PROJECT,
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


@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    account_input = (data.get('account') or '').strip()
    password_input = data.get('password') or ''

    if not account_input or not password_input:
        return jsonify({'success': False, 'message': '請輸入帳號與密碼'}), 400

    manager_account = os.getenv('MANAGER_ACCOUNT', '').strip()
    manager_hash = os.getenv('MANAGER_PASSWORD_HASH', '').strip()

    if not manager_account or not manager_hash:
        logging.error('MANAGER_ACCOUNT or MANAGER_PASSWORD_HASH not configured')
        return jsonify({'success': False, 'message': '伺服器尚未設定管理員帳號，請聯絡系統管理員'}), 500

    if account_input != manager_account or not check_password_hash(manager_hash, password_input):
        logging.warning("Login failed for account '%s' from %s", account_input, request.remote_addr)
        return jsonify({'success': False, 'message': '帳號或密碼錯誤'}), 401

    access, refresh = _generate_tokens(account_input)
    resp = make_response(jsonify({
        'success': True,
        'access_token': access,
        'user': {'account': account_input, 'permission': 'manager'},
    }), 200)
    _set_refresh_cookie(resp, refresh)
    return resp


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
        access, new_refresh = _generate_tokens(account)
        resp = make_response(jsonify({
            'success': True,
            'access_token': access,
            'user': {'account': account, 'permission': 'manager'},
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
