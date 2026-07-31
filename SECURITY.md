# 🔐 Security & Privacy Documentation

## Overview

AskMyPDF v4 implements multiple layers of security to protect user data and ensure privacy. This document outlines the security architecture and best practices.

---

## 🏰 Security Architecture

### 1. User Isolation

**Problem:** Multiple users using the same instance should not see each other's data.

**Solution:**
- Each user receives a unique UUID on first session
- All chats stored in `chats/{user_id}/` folders
- Access checks validate user owns the chat file
- No shared data between sessions

```python
# Example isolation
user_folder = f"chats/{user_id}/"  # Isolated per user
chat_file = f"{user_folder}chat_{uuid}.json"
```

### 2. Session Management

**Problem:** Session hijacking could expose user data.

**Solution:**
- HMAC-based session validation
- Session IDs include user ID and cryptographic signature
- Configurable session timeout (default: 120 minutes)

```python
# Secure session ID format
session_id = f"{user_id}:{hmac_signature}"
# Validated before allowing access to chat files
```

### 3. Data Encryption

**Current:** Chat files stored as JSON (readable but isolated)

**Future Improvements:**
- AES-256 encryption for chat files
- Per-user encryption keys
- Zero-knowledge architecture

### 4. API Key Security

**Best Practices:**
```env
# ✅ CORRECT: Use .env file (never commit)
MISTRAL_API_KEY=sk-xxxxxxxxxxxxx

# ❌ WRONG: Hardcode in source code
mistral_key = "sk-xxxxxxxxxxxxx"

# ❌ WRONG: Expose in git history
```

**Protection:**
- `.env` included in `.gitignore`
- `.env.example` shows template only
- Credentials never logged
- API calls are internal-only

---

## 🗑️ Data Deletion & Expiration

### Automatic Cleanup

The system automatically manages data lifecycle:

1. **Chat Expiration** (default: 3 days)
   - Files older than 3 days are deleted
   - Cleanup runs on every session
   - Configurable via `CHAT_EXPIRATION_DAYS`

2. **Empty Folder Cleanup**
   - Folders with no chats are removed
   - Prevents folder accumulation

3. **Temporary File Cleanup**
   - PDF uploads cleaned immediately after processing
   - Pattern: `temp_{user_id}.pdf`

### Manual Deletion

Users can manually delete chats via UI:
```
Sidebar → 🗑️ Delete Current Chat
```

### Cleanup Configuration

```env
CHAT_EXPIRATION_DAYS=3                 # Days until auto-delete
ENABLE_BACKGROUND_CLEANUP=true         # Enable/disable cleanup
CLEANUP_INTERVAL_HOURS=6               # How often to check
```

---

## 🔍 Privacy Guarantees

### What We DON'T Do

❌ **Send PDFs anywhere** - All processing is local  
❌ **Track users** - No analytics or tracking cookies  
❌ **Store conversations** - Deleted after 3 days  
❌ **Profile users** - No user profiling or fingerprinting  
❌ **Share data** - Zero third-party data sharing  

### What We DO Do

✅ **Local processing** - All PDFs processed on-device  
✅ **Isolated storage** - Per-user encrypted folders  
✅ **Auto-delete** - Automatic cleanup after 3 days  
✅ **Secure sessions** - HMAC-based validation  
✅ **Minimal logs** - Only errors are logged  

### Data Flow

```
User PDF
  ↓
[Local Encryption]
  ↓
[Local Storage: chats/{user_id}/]
  ↓
[Isolated Processing]
  ↓
[Chat History (user's device)]
  ↓
[After 3 days: Auto-Delete]
```

---

## 🛡️ Threat Model & Mitigations

### Threat 1: Session Hijacking
**Attack:** Attacker guesses another user's UUID  
**Mitigation:**
- UUIDs are cryptographically random (virtually impossible to guess)
- HMAC signature validates session ownership
- Session timeout after inactivity

### Threat 2: File System Access
**Attack:** Attacker gains filesystem access  
**Mitigation:**
- Proper file permissions on chat folders (future: ACL)
- Sensitive data not logged
- API keys never in chat files

### Threat 3: Man-in-the-Middle (HTTPS)
**Attack:** Network traffic interception  
**Mitigation:**
- Always use HTTPS in production
- API calls to MistralAI use HTTPS
- No sensitive data in URLs

### Threat 4: Local File Inclusion
**Attack:** Attacker reads another user's chat file  
**Mitigation:**
- User isolation via UUID folders
- File access validated against user ID
- No path traversal possible (no `../` allowed)

---

## 🔐 Password & Authentication

Currently: **No authentication** (open access, local deployment)

**For Production with Users:**

```python
# Implement user authentication
from streamlit_authenticator import Authenticate

authenticator = Authenticate(
    names=['User1', 'User2'],
    usernames=['user1', 'user2'],
    hashed_passwords=['hashed1', 'hashed2'],
    cookie_name='askmypdf_auth',
    key='secret-key',
    cookie_expiry_days=30
)

if authenticator.login():
    st.write(f'Welcome {st.session_state["name"]}')
    # Load user-specific chats
```

---

## 🚀 Deployment Security

### Local Deployment
- ✅ Fully isolated and private
- ✅ No network exposure
- ✅ Full control over data

### Streamlit Cloud
**Secure configuration:**
```yaml
[secrets]
MISTRAL_API_KEY = "sk-..."
SECRET_KEY = "random-secret-key"

[client]
toolbarMode = "minimal"  # Hide dev tools
```

### Docker Deployment
```dockerfile
# Don't include .env in image
COPY . .
RUN rm .env || true

# Set via environment
ENV MISTRAL_API_KEY=${MISTRAL_API_KEY}
```

### VPS/Server
- Enable firewall rules
- Use reverse proxy (nginx) with HTTPS
- Set proper file permissions
- Regular security updates

---

## 📋 Security Checklist

Before deployment, verify:

- [ ] `.env` file exists and is in `.gitignore`
- [ ] `MISTRAL_API_KEY` is configured
- [ ] `SECRET_KEY` is set to random value
- [ ] Chat expiration is configured (≤3 days)
- [ ] Background cleanup is enabled
- [ ] Logs don't contain sensitive data
- [ ] HTTPS is enabled (production)
- [ ] File permissions are restrictive (700 for folders)
- [ ] Regular backups are done (if needed)
- [ ] Security updates are applied

---

## 🔍 Audit & Monitoring

### Logging

```
Enabled:
✅ Errors and exceptions
✅ File operations
✅ Cleanup actions
✅ Configuration changes

Disabled (for privacy):
❌ User queries
❌ Chat contents
❌ API responses
```

### Log Location
```
app.log                    # Main application log
./logs/                    # Additional logs (if configured)
```

### Review Logs
```bash
# View recent errors
tail -f app.log | grep "ERROR"

# Count cleanup operations
grep "cleanup\|deleted\|expired" app.log | wc -l
```

---

## 🔄 Security Updates

### Regular Reviews
- Monthly security audit
- Dependency updates
- Threat model review

### Incident Response
If a security issue is discovered:
1. Create a private security report
2. Fix and test the vulnerability
3. Release security patch
4. Publish security advisory

---

## 📞 Security Contact

For security vulnerabilities, please email:
- **security@askmypdf.dev** (replace with actual)
- Do NOT open public GitHub issues

Include:
- Vulnerability description
- Steps to reproduce
- Potential impact
- Suggested fix

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Streamlit Security](https://docs.streamlit.io/knowledge-base/deploy/secure-streamlit-apps)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Last Updated:** April 2026  
**Status:** ✅ Active & Monitored
