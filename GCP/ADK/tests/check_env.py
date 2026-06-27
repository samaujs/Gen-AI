import os
for k, v in os.environ.items():
    if "git" in k.lower() or "token" in k.lower() or "pass" in k.lower() or "key" in k.lower() or "auth" in k.lower():
        # Redact the actual value
        print(f"{k}: {'***' if v else 'empty'}")
