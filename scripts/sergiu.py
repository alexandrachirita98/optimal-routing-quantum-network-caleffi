import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.handlers.figures import Figures
from utils.models.physical_params import PhysicalParams
from utils.models.optimal_routing import OptimalRouting
from utils.models.topology import Topology

figs = Figures()
figs.figure9_paper_faithful()