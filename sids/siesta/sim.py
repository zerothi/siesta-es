"""
Simulation wrapper to find files belonging to SIESTA
"""

# Import the simulation module for obtaining the data
import sids.simulation as _sim
import sids.siesta.files as _files

# Create a list of rules that SIESTA files obeys
rules = []
rules.append(_sim.RuleFile(obj=_files.HSX,ext='HSX',type='HSX'))
rules.append(_sim.RuleFile(obj=_files.HS,ext='HS',type='HS'))
rules.append(_sim.RuleFile(obj=_files.TSHS,ext='TSHS',type='TSHS'))
rules.append(_sim.RuleFile(obj=_files.DM,ext='DM',type='DM'))

