import cProfile
from conehead.conehead import run

pr = cProfile.Profile()
pr.enable()

run()

pr.disable()
pr.print_stats()
