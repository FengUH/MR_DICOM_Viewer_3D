"""
Minimal imghdr stub for Python 3.13+ to prevent older Streamlit versions
that still `import imghdr` from crashing.

This app does not rely on imghdr for any functionality, so it's safe
to always return None and allow Pillow or other imaging libraries
to handle image type detection.
"""

def what(file, h=None):
    # Placeholder implementation matching the standard library signature.
    # Always return None.
    return None
