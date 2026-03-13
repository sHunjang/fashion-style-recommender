# app.py (수정)

"""
HuggingFace Spaces 배포용 진입점
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# app.py → app/demo.py 직접 import 불가!
# 대신 demo.py 내용을 직접 실행!
import importlib.util

spec = importlib.util.spec_from_file_location(
    "demo",
    Path(__file__).parent / "app" / "demo.py"
)
demo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_module)

demo = demo_module.demo

if __name__ == "__main__":
    demo.launch()