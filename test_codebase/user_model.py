
# User model for authentication system
class User:
    def __init__(self, user_id: str, username: str, email: str):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.is_active = True
        self.login_attempts = 0
    
    def deactivate(self):
        """Deactivate a user account"""
        self.is_active = False
        return True
    
    def increment_login_attempts(self):
        """Increment the number of login attempts"""
        self.login_attempts += 1
        if self.login_attempts >= 3:
            self.deactivate()
        return self.login_attempts
    
    @property
    def display_name(self):
        """Get user's display name"""
        return self.username or self.email.split('@')[0]
    
    def __str__(self):
        return f"User({self.username}, {self.email})"
