from core.swarm import Swarm
from configs.formations import create_cube_formation


swarm = Swarm(64)
create_cube_formation(swarm, size=4)

test = swarm.get_all_cubes()
print(test)