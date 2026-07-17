"""测试3D conformer特征修复"""
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import logging
logging.disable()

# test with beta-caryophyllene (HERB SMILES)
smi = 'CC1=CCCC(=C)C2CC(C2CC1)(C)C'

print(f"Testing SMILES: {smi}")

mol = Chem.MolFromSmiles(smi)
if mol is None:
    print("FAIL: can't parse SMILES")
    exit(1)

mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42

s = AllChem.EmbedMolecule(mol, params)
print(f"EmbedMolecule status: {s} (0=success)")

ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
print(f"ForceField: {ff is not None}")

if ff:
    e = ff.CalcEnergy()
    print(f"Initial Energy: {e:.2f}")
    
    s2 = AllChem.MMFFOptimizeMolecule(mol)
    print(f"MMFFOptimizeMolecule status: {s2} (0=success)")
    
    e2 = ff.CalcEnergy()
    print(f"Optimized Energy: {e2:.2f}")
    
    pmi1 = Descriptors.PMI1(mol)
    pmi2 = Descriptors.PMI2(mol)
    pmi3 = Descriptors.PMI3(mol)
    print(f"PMI: {pmi1:.2f}, {pmi2:.2f}, {pmi3:.2f}")
    
    npr1 = Descriptors.NPR1(mol)
    npr2 = Descriptors.NPR2(mol)
    print(f"NPR: {npr1:.4f}, {npr2:.4f}")
    
    print("\n3D Conformer Computation: SUCCESS!")
else:
    print("FAIL: could not get MMFF force field")
