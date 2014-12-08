"""
Simulation wrapper to find files belonging to SIESTA
"""

# Import the simulation module for obtaining the data
import sids.simulation as _sim
import sids.siesta.files as _files
import sids.siesta.io as _io

# Create a list of rules that SIESTA files obeys
rules = []
rules.append(_sim.RuleFile(obj=_io.XV,ext='XV',type='XV',
                           variables=['cell','xa']))
rules.append(_sim.RuleFile(obj=_io.FA,ext='FA',type='FA',
                           variables=['F']))
rules.append(_sim.RuleFile(obj=_files.TSHS,ext='TSHS',type='TSHS'))
rules.append(_sim.RuleFile(obj=_files.HSX,ext='HSX',type='HSX'))
rules.append(_sim.RuleFile(obj=_files.HS,ext='HS',type='HS'))
rules.append(_sim.RuleFile(obj=_files.SE_TSHS,ext='TSHS',type='SE_TSHS'))
rules.append(_sim.RuleFile(obj=_files.SE_HSX,ext='HSX',type='SE_HSX'))
rules.append(_sim.RuleFile(obj=_files.SE_HS,ext='HS',type='SE_HS'))
rules.append(_sim.RuleFile(obj=_files.DM,ext='DM',type='DM'))
rules.append(_sim.RuleFile(obj=_io.ANI,ext='ANI',type='ANI'))
rules.append(_sim.RuleFile(obj=_io.XYZ,ext='xyz',type='XYZ'))
rules.append(_sim.RuleFile(obj=_io.TRANS,ext='TRANS',type='TRANS'))
rules.append(_sim.RuleFile(obj=_io.TRANS,ext='AVTRANS',type='AVTRANS'))
