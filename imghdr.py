"""
Minimal imghdr stub for Python 3.13+ so that older Streamlit versions
that still `import imghdr` won't crash.

We don't actually rely on imghdr anywhere in this app, so it's safe
to always return None and let Pillow / other libs handle image types.
"""

def what(file, h=None):
    # 按标准库签名留一个占位实现，返回 None 即可
    return None
