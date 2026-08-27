# Run the session setup headlessly and freeze the result, for reproducing a session
# that has already been approved on screen.
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

runpy.run_path(HERE + "/locres_gui_session.py",
               init_globals={"session": session}, run_name="__main__")
runpy.run_path(HERE + "/save_poses.py",
               init_globals={"session": session}, run_name="__main__")
from chimerax.core.commands import run
run(session, "exit")
