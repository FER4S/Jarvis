# ─────────────────────────────────────────────────────────────────────────────
#  core/email_accounts.py – Encrypted local store of connected email accounts
#  Accounts (IMAP host/user/password, Gmail OAuth refresh tokens) are
#  encrypted at rest with Windows DPAPI (CryptProtectData) - no key file
#  exists anywhere, so OneDrive syncing data/ only ever moves ciphertext.
#
#  Threat model: DPAPI ties the ciphertext to this Windows user account on
#  this machine (like a browser's saved-password store) - any process running
#  as this same user can decrypt it; there is no additional secret to steal or
#  leak. Moving the encrypted file to another machine/user (e.g. via OneDrive
#  sync) makes it permanently undecryptable there - load() quarantines it and
#  starts empty rather than crash (see _quarantine_unreadable_file).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import ctypes
import json
import os
import sys
import threading
import time
import uuid
from ctypes import wintypes
from datetime import datetime

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/email_accounts.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger

import config

if os.name != "nt":
    raise RuntimeError(
        "core.email_accounts requires Windows (it encrypts credentials via "
        "crypt32.dll's DPAPI, which has no equivalent on other platforms)."
    )

# ── Storage location ─────────────────────────────────────────────────────────
_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")
_ACCOUNTS_FILE: str = os.path.join(_DATA_DIR, "email_accounts.dat")

# Read attempts before declaring the store unreadable and quarantining it.
# OneDrive/antivirus can hold a transient lock on the file; a short retry
# rides that out instead of treating it as corruption. (Mirrors
# core/memory.py's _LOAD_ATTEMPTS/_LOAD_RETRY_DELAY_S exactly.)
_LOAD_ATTEMPTS: int = 3
_LOAD_RETRY_DELAY_S: float = 0.25

STORE_VERSION: int = 1
PROVIDER_IMAP: str = "imap"
PROVIDER_GMAIL: str = "gmail_oauth"

# Fields exposed by list_safe()/the safe view returned from add_*() - never
# "credentials". Adding a new provider never needs to touch this.
_SAFE_FIELDS: tuple[str, ...] = ("id", "label", "provider", "created_at")


# ── DPAPI (Windows Data Protection API) via ctypes ───────────────────────────
# No third-party crypto dependency: crypt32.dll/CryptProtectData is stdlib-
# reachable via ctypes and ties the ciphertext to this Windows user account
# with no key file to manage or leak.

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_crypt32.CryptProtectData.argtypes = [
    ctypes.POINTER(_DATA_BLOB), wintypes.LPCWSTR, ctypes.POINTER(_DATA_BLOB),
    wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_DATA_BLOB),
]
_crypt32.CryptProtectData.restype = wintypes.BOOL

_crypt32.CryptUnprotectData.argtypes = [
    ctypes.POINTER(_DATA_BLOB), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DATA_BLOB),
    wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(_DATA_BLOB),
]
_crypt32.CryptUnprotectData.restype = wintypes.BOOL

_kernel32.LocalFree.argtypes = [wintypes.LPVOID]
_kernel32.LocalFree.restype = wintypes.LPVOID

# Never let DPAPI show a UI prompt - this process is headless and a blocked
# prompt would hang the caller (a request thread or the poll thread) forever.
_CRYPTPROTECT_UI_FORBIDDEN = 0x01


def _to_blob(data: bytes) -> tuple[_DATA_BLOB, ctypes.Array]:
    """
    Build a DATA_BLOB pointing at a live ctypes buffer. The buffer is returned
    alongside the blob so the caller keeps a reference to it for the duration
    of the call - the blob only holds a raw pointer into it, so letting the
    buffer get garbage-collected first would leave a dangling pointer.
    """
    buf = ctypes.create_string_buffer(data, len(data))
    blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def dpapi_encrypt(plaintext: bytes) -> bytes:
    """Encrypt bytes with CryptProtectData, scoped to this Windows user account."""
    in_blob, _keepalive = _to_blob(plaintext)
    out_blob = _DATA_BLOB()
    ok = _crypt32.CryptProtectData(
        ctypes.byref(in_blob), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(f"CryptProtectData failed (Win32 error {ctypes.get_last_error()})")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        _kernel32.LocalFree(out_blob.pbData)


def dpapi_decrypt(ciphertext: bytes) -> bytes:
    """
    Decrypt bytes produced by dpapi_encrypt(). Raises OSError if this isn't
    the same Windows user account/machine that encrypted it, or the data is
    corrupt/tampered - callers must treat that as unrecoverable, not retry.
    """
    in_blob, _keepalive = _to_blob(ciphertext)
    out_blob = _DATA_BLOB()
    ok = _crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, None, None, None,
        _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(f"CryptUnprotectData failed (Win32 error {ctypes.get_last_error()})")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        _kernel32.LocalFree(out_blob.pbData)


class EmailAccountStore:
    """
    DPAPI-encrypted local store of connected email accounts.

    Mirrors core.memory.MemoryManager's persistence pattern: a single lock,
    atomic tmp-file + os.replace writes, and quarantine-on-unreadable rather
    than ever overwriting the boss's saved accounts with an empty fallback
    state. The only difference is the file holds DPAPI ciphertext instead of
    plaintext JSON.

    Adding a future provider type (e.g. Outlook OAuth) needs only a new
    PROVIDER_* string and its own credentials dict shape - no schema rewrite,
    since "credentials" is opaque to the store itself.

    Usage:
        store = EmailAccountStore()
        store.load()
        store.add_imap_account("Hostinger Support", "imap.hostinger.com", 993, "u", "p")
    """

    def __init__(self, path: str = _ACCOUNTS_FILE) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._accounts: list[dict] = []
        # Set when the store file was unreadable AND couldn't be quarantined -
        # every save is then refused so it's never overwritten by this
        # session's empty fallback state.
        self._save_disabled = False

    # ── Storage ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load accounts from disk, creating an empty encrypted store if missing.

        If the file exists but can't be read/decrypted/parsed even after a
        short retry (transient OneDrive/AV locks, or - permanently - ciphertext
        that DPAPI on this machine/user can never decrypt, e.g. after a
        OneDrive sync to a different PC), it is quarantined: renamed aside for
        manual recovery, never overwritten. The boss re-adds accounts via the
        settings UI; nothing crashes.
        """
        with self._lock:
            if not os.path.isdir(_DATA_DIR):
                os.makedirs(_DATA_DIR, exist_ok=True)

            if not os.path.isfile(self._path):
                logger.info(f"No email account store found at {self._path} - creating a new one.")
                self._accounts = []
                self._write_locked()
                return

            last_exc: Exception | None = None
            for attempt in range(1, _LOAD_ATTEMPTS + 1):
                # ValueError covers json.JSONDecodeError plus the wrong-shape
                # check below; OSError covers locked/unreadable files AND a
                # DPAPI decrypt failure (dpapi_decrypt raises OSError).
                try:
                    with open(self._path, "rb") as f:
                        ciphertext = f.read()
                    plaintext = dpapi_decrypt(ciphertext)
                    loaded = json.loads(plaintext.decode("utf-8"))
                    if not isinstance(loaded, dict) or not isinstance(loaded.get("accounts"), list):
                        raise ValueError("unexpected store shape")
                    self._accounts = loaded["accounts"]
                    logger.success(f"Loaded {len(self._accounts)} email account(s) from {self._path}.")
                    return
                except (ValueError, OSError) as exc:
                    last_exc = exc
                    if attempt < _LOAD_ATTEMPTS:
                        logger.warning(
                            f"Failed to load email account store (attempt {attempt}/"
                            f"{_LOAD_ATTEMPTS}: {exc}) - retrying in {_LOAD_RETRY_DELAY_S}s…"
                        )
                        time.sleep(_LOAD_RETRY_DELAY_S)

            self._quarantine_unreadable_file(last_exc)

    def _quarantine_unreadable_file(self, exc: Exception | None) -> None:
        """
        Handle a store file that exists but stayed unreadable through the
        retry loop. Caller must already hold self._lock.

        Same rationale as MemoryManager._quarantine_unreadable_file(): never
        let a later save() overwrite real accounts with this session's empty
        fallback. This also covers the DPAPI-specific permanent case
        (ciphertext encrypted under a different Windows user/machine can never
        be decrypted here) - there is no way to recover it, only to preserve
        it for inspection and start fresh.
        """
        timestamp = datetime.now(config.TIMEZONE).strftime("%Y%m%d-%H%M%S")
        quarantine_path = f"{self._path}.corrupt-{timestamp}"
        try:
            os.replace(self._path, quarantine_path)
            logger.critical(
                f"Email account store could not be loaded ({exc}) - moved it to "
                f"{quarantine_path} for manual recovery. Starting with NO email "
                f"accounts connected; re-add them from the settings UI."
            )
        except OSError as move_exc:
            self._save_disabled = True
            logger.critical(
                f"Email account store could not be loaded ({exc}) and could not "
                f"be quarantined ({move_exc}) - saves are DISABLED for this "
                f"session to protect the file on disk. Restart Jarvis once the "
                f"file is accessible again."
            )
        self._accounts = []

    def _write_locked(self) -> None:
        """
        Actual write logic. Caller must already hold self._lock. Only ever
        logs account counts, never labels/hosts/usernames/secrets.
        """
        if self._save_disabled:
            logger.error(
                "Email account store save skipped - saves are disabled for "
                "this session after an unreadable store was left in place "
                "(see startup log)."
            )
            return
        os.makedirs(_DATA_DIR, exist_ok=True)
        payload = {"version": STORE_VERSION, "accounts": self._accounts}
        plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            ciphertext = dpapi_encrypt(plaintext)
        except OSError as exc:
            logger.error(f"Failed to encrypt email account store: {exc}")
            return
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(ciphertext)
            os.replace(tmp_path, self._path)  # atomic on both Windows and POSIX
        except OSError as exc:
            logger.error(f"Failed to save email account store: {exc}")

    # ── Accounts ──────────────────────────────────────────────────────────────

    def add_imap_account(
        self,
        label: str,
        host: str,
        port: int,
        username: str,
        password: str,
        use_ssl: bool = True,
    ) -> dict:
        """
        Add a new IMAP account and save immediately. Returns the SAFE view
        (no credentials) - callers needing the credentials back should use
        get()/list_all() instead (the fetch/poll path only).
        """
        account = {
            "id": uuid.uuid4().hex,
            "label": label.strip() or host,
            "provider": PROVIDER_IMAP,
            "created_at": datetime.now(config.TIMEZONE).isoformat(),
            "credentials": {
                "host": host.strip(),
                "port": int(port),
                "username": username.strip(),
                "password": password,
                "use_ssl": bool(use_ssl),
            },
        }
        with self._lock:
            self._accounts.append(account)
            self._write_locked()
        logger.success(f"IMAP email account added: '{account['label']}' (host {host}) - id {account['id']}")
        return self._safe_view(account)

    def add_gmail_account(self, label: str, refresh_token: str, email_address: str) -> dict:
        """
        Add a new Gmail OAuth account and save immediately. Returns the SAFE
        view (no refresh token).
        """
        account = {
            "id": uuid.uuid4().hex,
            "label": label.strip() or email_address,
            "provider": PROVIDER_GMAIL,
            "created_at": datetime.now(config.TIMEZONE).isoformat(),
            "credentials": {
                "refresh_token": refresh_token,
                "email_address": email_address.strip(),
            },
        }
        with self._lock:
            self._accounts.append(account)
            self._write_locked()
        logger.success(f"Gmail email account added: '{account['label']}' - id {account['id']}")
        return self._safe_view(account)

    def list_safe(self) -> list[dict]:
        """All accounts with credentials stripped - safe for an API response."""
        with self._lock:
            return [self._safe_view(a) for a in self._accounts]

    def list_all(self) -> list[dict]:
        """
        Deep copies of every account INCLUDING credentials. For the fetch/
        poll path only - never expose this to an API response.
        """
        with self._lock:
            return copy.deepcopy(self._accounts)

    def get(self, account_id: str) -> dict | None:
        """
        Deep copy of one account INCLUDING credentials, or None if not found.
        For the fetch/poll path only.
        """
        with self._lock:
            for account in self._accounts:
                if account.get("id") == account_id:
                    return copy.deepcopy(account)
        return None

    def delete(self, account_id: str) -> bool:
        """Remove an account by id. Returns True if it existed."""
        with self._lock:
            before = len(self._accounts)
            self._accounts = [a for a in self._accounts if a.get("id") != account_id]
            removed = len(self._accounts) != before
            if removed:
                self._write_locked()
        if removed:
            logger.success(f"Email account removed: id {account_id}")
        return removed

    @staticmethod
    def _safe_view(account: dict) -> dict:
        return {field: account.get(field) for field in _SAFE_FIELDS}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    print("\n--- DPAPI round-trip ---")
    _plaintext = b"jarvis-dpapi-smoke-test"
    _ciphertext = dpapi_encrypt(_plaintext)
    assert _ciphertext != _plaintext
    _decrypted = dpapi_decrypt(_ciphertext)
    assert _decrypted == _plaintext
    print("Round-trip OK: plaintext survives encrypt -> decrypt unchanged.")

    print("\n--- Tamper detection ---")
    _tampered = bytearray(_ciphertext)
    _tampered[-1] ^= 0xFF
    try:
        dpapi_decrypt(bytes(_tampered))
        print("UNEXPECTED: tampered ciphertext decrypted without error!")
    except OSError as exc:
        print(f"Tampered ciphertext correctly rejected: {exc}")

    print("\n--- Store smoke ---")
    store = EmailAccountStore()
    store.load()
    print(f"Accounts on disk: {len(store.list_safe())}")

    smoke_account = store.add_imap_account(
        label="__smoke__",
        host="imap.example.com",
        port=993,
        username="smoke@example.com",
        password="not-a-real-password",
    )
    print(f"Added smoke account: {smoke_account}")
    print(f"Safe list now: {store.list_safe()}")
    assert store.delete(smoke_account["id"])
    print("Smoke account deleted.")
    print(f"Safe list after delete: {store.list_safe()}")
