from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("app_user_tender.py")), run_name="__main__")
