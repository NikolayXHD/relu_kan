import base64
from hashlib import sha1


def sha1_hash(value: str) -> str:
    """hash string of 40 hex characters"""
    h = sha1()
    h.update(value.encode('utf-8'))
    return h.hexdigest()


def sha1_hash_base64(value: str) -> str:
    """hash string of 28 base64 characters"""
    h = sha1()
    h.update(value.encode('utf-8'))
    return base64.b64encode(h.digest()).decode('utf-8')


__all__ = ['sha1_hash', 'sha1_hash_base64']
