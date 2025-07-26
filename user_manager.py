import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import uuid
import jwt
from jwt.exceptions import InvalidTokenError
import yaml

LOGGER = logging.getLogger("aks")

class UserManager:
    """
    Comprehensive user management system for AKS with:
    - Secure authentication
    - Role-based access control
    - API key management
    - User activity tracking
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize UserManager with configuration.
        
        Args:
            config_path: Path to user configuration directory
        """
        self.config_path = config_path.resolve()
        self.users_file = self.config_path / "users.json"
        self.roles_file = self.config_path / "roles.yaml"
        self.sessions = {}
        self.jwt_secret = secrets.token_hex(32)
        self.token_expiry = timedelta(hours=12)
        
        # Create directories if they don't exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.users = {}
        self.roles = {
            "admin": ["*"],
            "editor": ["read", "write", "execute"],
            "viewer": ["read"]
        }
        
        self._load_data()
        LOGGER.info("UserManager initialized")

    def _load_data(self):
        """Load user and role data from files."""
        try:
            # Load users
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            
            # Load roles
            if self.roles_file.exists():
                with open(self.roles_file, 'r') as f:
                    self.roles = yaml.safe_load(f) or self.roles
        except Exception as e:
            LOGGER.error(f"Failed to load user data: {e}")
            raise RuntimeError("Could not initialize user data") from e

    def _save_data(self):
        """Persist user and role data to files."""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            
            with open(self.roles_file, 'w') as f:
                yaml.dump(self.roles, f)
        except Exception as e:
            LOGGER.error(f"Failed to save user data: {e}")

    def create_user(self, username: str, password: str, roles: List[str], 
                  email: str = "", full_name: str = "") -> bool:
        """
        Create a new user with secure password storage.
        
        Args:
            username: Unique username
            password: Plaintext password (will be hashed)
            roles: List of role names
            email: User email (optional)
            full_name: User's full name (optional)
            
        Returns:
            bool: True if user was created successfully
        """
        if username in self.users:
            LOGGER.warning(f"User {username} already exists")
            return False
            
        if not self._validate_password(password):
            LOGGER.warning("Invalid password requirements")
            return False
            
        # Hash password with salt
        salt = secrets.token_hex(16)
        hashed_pw = self._hash_password(password, salt)
        
        # Create user record
        self.users[username] = {
            "password_hash": hashed_pw,
            "salt": salt,
            "roles": roles,
            "email": email,
            "full_name": full_name,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "api_keys": {},
            "is_active": True
        }
        
        self._save_data()
        LOGGER.info(f"Created user {username}")
        return True

    def _validate_password(self, password: str) -> bool:
        """Enforce password policy."""
        if len(password) < 10:
            return False
        if not any(c.isupper() for c in password):
            return False
        if not any(c.isdigit() for c in password):
            return False
        return True

    def _hash_password(self, password: str, salt: str) -> str:
        """Generate secure password hash."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()

    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate a user and return JWT token if successful.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            Tuple of (success, token_or_error_message)
        """
        user = self.users.get(username)
        if not user or not user.get("is_active", True):
            LOGGER.warning(f"Authentication failed for {username}: user not found or inactive")
            return False, "Invalid credentials"
            
        # Verify password
        hashed_pw = self._hash_password(password, user["salt"])
        if hashed_pw != user["password_hash"]:
            LOGGER.warning(f"Authentication failed for {username}: invalid password")
            return False, "Invalid credentials"
            
        # Update last login
        user["last_login"] = datetime.utcnow().isoformat()
        self._save_data()
        
        # Generate JWT token
        token = self._generate_jwt(username)
        LOGGER.info(f"User {username} authenticated successfully")
        return True, token

    def _generate_jwt(self, username: str) -> str:
        """Generate JWT token for authenticated user."""
        user = self.users[username]
        payload = {
            "sub": username,
            "roles": user["roles"],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Verify JWT token and return user claims if valid.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Tuple of (is_valid, claims_or_error)
        """
        try:
            claims = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            if claims["sub"] not in self.users:
                raise InvalidTokenError("User not found")
            return True, claims
        except InvalidTokenError as e:
            LOGGER.warning(f"Token verification failed: {str(e)}")
            return False, str(e)

    def create_api_key(self, username: str, key_name: str) -> Tuple[bool, Optional[str]]:
        """
        Generate a new API key for the user.
        
        Args:
            username: User to create key for
            key_name: Descriptive name for the key
            
        Returns:
            Tuple of (success, key_or_error_message)
        """
        if username not in self.users:
            return False, "User not found"
            
        api_key = secrets.token_urlsafe(32)
        key_id = str(uuid.uuid4())
        
        self.users[username]["api_keys"][key_id] = {
            "name": key_name,
            "key": hashlib.sha256(api_key.encode()).hexdigest(),
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "is_active": True
        }
        
        self._save_data()
        LOGGER.info(f"Created API key '{key_name}' for {username}")
        return True, api_key

    def verify_api_key(self, username: str, api_key: str) -> bool:
        """
        Verify if an API key is valid for the given user.
        
        Args:
            username: User to verify
            api_key: API key to check
            
        Returns:
            bool: True if key is valid
        """
        if username not in self.users:
            return False
            
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_data in self.users[username]["api_keys"].values():
            if key_data["key"] == hashed_key and key_data["is_active"]:
                key_data["last_used"] = datetime.utcnow().isoformat()
                self._save_data()
                return True
                
        return False

    def has_permission(self, username: str, permission: str) -> bool:
        """
        Check if user has the specified permission.
        
        Args:
            username: User to check
            permission: Permission string (e.g. "read", "write")
            
        Returns:
            bool: True if user has permission
        """
        if username not in self.users:
            return False
            
        user_roles = self.users[username]["roles"]
        
        for role in user_roles:
            if role in self.roles:
                if "*" in self.roles[role] or permission in self.roles[role]:
                    return True
                    
        return False

    def get_user_activity(self, username: str) -> Dict:
        """
        Get user activity information.
        
        Args:
            username: User to query
            
        Returns:
            Dict containing activity information
        """
        if username not in self.users:
            return {}
            
        return {
            "last_login": self.users[username].get("last_login"),
            "api_keys": {
                k: {"name": v["name"], "last_used": v.get("last_used")}
                for k, v in self.users[username]["api_keys"].items()
            }
        }

    def revoke_api_key(self, username: str, key_id: str) -> bool:
        """
        Revoke/disable an API key.
        
        Args:
            username: User owning the key
            key_id: ID of key to revoke
            
        Returns:
            bool: True if key was revoked
        """
        if username not in self.users:
            return False
            
        if key_id not in self.users[username]["api_keys"]:
            return False
            
        self.users[username]["api_keys"][key_id]["is_active"] = False
        self._save_data()
        LOGGER.info(f"Revoked API key {key_id} for {username}")
        return True

    def update_user_roles(self, username: str, roles: List[str]) -> bool:
        """
        Update a user's roles.
        
        Args:
            username: User to update
            roles: New list of roles
            
        Returns:
            bool: True if roles were updated
        """
        if username not in self.users:
            return False
            
        # Verify all roles exist
        for role in roles:
            if role not in self.roles:
                return False
                
        self.users[username]["roles"] = roles
        self._save_data()
        LOGGER.info(f"Updated roles for {username}: {roles}")
        return True

    def list_users(self) -> List[Dict]:
        """
        Get list of all users (without sensitive data).
        
        Returns:
            List of user dictionaries
        """
        return [
            {
                "username": uname,
                "email": data["email"],
                "full_name": data["full_name"],
                "roles": data["roles"],
                "created_at": data["created_at"],
                "is_active": data["is_active"]
            }
            for uname, data in self.users.items()
        ]

    def deactivate_user(self, username: str) -> bool:
        """
        Deactivate a user account.
        
        Args:
            username: User to deactivate
            
        Returns:
            bool: True if user was deactivated
        """
        if username not in self.users:
            return False
            
        self.users[username]["is_active"] = False
        self._save_data()
        LOGGER.info(f"Deactivated user {username}")
        return True

    def rotate_jwt_secret(self):
        """Rotate the JWT signing secret (invalidates all existing tokens)."""
        self.jwt_secret = secrets.token_hex(32)
        LOGGER.warning("Rotated JWT secret - all existing tokens will be invalid")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure data is saved."""
        self._save_data()
