
# Authentication service
from typing import Dict, Optional, Tuple
import hashlib
import secrets
from datetime import datetime, timedelta

# Import from another file in the project
from user_model import User

class AuthService:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.users = {}  # user_id -> User
        self.tokens = {}  # token -> user_id
    
    def _load_config(self, config_path: str) -> Dict:
        """Load auth service configuration"""
        # This should call data_processor.load_data but we're simplifying
        # Missing dependency that might cause confusion
        return {"token_expiry_hours": 24, "hash_algorithm": "sha256"}
    
    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """Register a new user"""
        # Check if user exists
        for user in self.users.values():
            if user.email == email:
                return False, "Email already registered"
        
        # Create user
        user_id = secrets.token_hex(8)
        password_hash = self._hash_password(password)
        
        # Save user
        user = User(user_id, username, email)
        self.users[user_id] = user
        
        # Missing: Save password hash to storage
        
        return True, user_id
    
    def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate a user and return a token"""
        # Find user by email
        user_id = None
        for uid, user in self.users.items():
            if user.email == email:
                user_id = uid
                break
        
        if not user_id:
            return None
        
        # Verify password - this would check against stored hash
        # For demo purposes, we're just returning a token
        
        # Generate token
        token = secrets.token_hex(16)
        expiry = datetime.now() + timedelta(hours=self.config["token_expiry_hours"])
        self.tokens[token] = {"user_id": user_id, "expiry": expiry}
        
        return token
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using the configured algorithm"""
        if self.config["hash_algorithm"] == "sha256":
            return hashlib.sha256(password.encode()).hexdigest()
        else:
            # Intentional bug: fallback should use a secure method
            return password  # Insecure!
    
    def validate_token(self, token: str) -> Optional[str]:
        """Validate a token and return the user_id if valid"""
        if token not in self.tokens:
            return None
        
        token_data = self.tokens[token]
        if token_data["expiry"] < datetime.now():
            # Remove expired token
            del self.tokens[token]
            return None
        
        return token_data["user_id"]
