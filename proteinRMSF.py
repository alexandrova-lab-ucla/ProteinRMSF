#!/usr/bin/env Python3

import sys
import os
import numpy as np
import math

BACKBONE_ATOMS = ["N", "CA", "C", "O"]
cwd = os.getcwd()

# Takes in a .pdb file and returns atoms as coords + name (tuple (string, coords))
# Coords are transformed to the center
def extractPDB(pdbFile):

    try:
        with open(pdbFile) as f:
            content = f.readlines()
    except IOError:
        print("The following file does not exist: ", pdbFile)
        sys.exit(0)

    # Removes all of the blank lines
    content = [x.strip() for x in content]

    if len(content) == 0:
        raise Exception("The file: ", pdbFile, " is empty. Cannot extract any meaningful data from this.")

    # A list of residues
    protein = []
    residueName = ""
    residueAtoms = []
    residueNumber = 0

    for line in content:
        line = line.split(" ")
        line = list(filter(None, line))
        if len(line) !=0 and line[0] == "ATOM":
            # Keep same residue
            if residueNumber != int(line[5]):
                if len(residueAtoms) != 0:
                    protein.append((residueName, residueNumber, residueAtoms))
                residueNumber += 1
                residueName = line[3]
                residueAtoms = []

            residueAtoms.append((line[2], np.array([float(line[6]), float(line[7]), float(line[8])])))

    # Always add the last one to the list!
    protein.append((residueName, residueNumber, residueAtoms))
    protein = center(protein)
    return protein


# Extracts the atoms in the protein, regardless of which residue they belong to
def extractAtomCoords(protein):
    atoms = []
    for residue in protein:
        for atom in residue[2]:
            atoms.append(atom)

    return atoms


# Extracts all of the QM atoms given from the qmList
def extractQMAtoms(protein, qmList):

    atoms = []

    for residue in qmList:
        for atom in protein[residue[1]][2]:
            atoms.append(atom)
    return atoms


# Extracts all of the alpha carbons in the protein
def extractAlphaCarbons(protein):
    atoms = []
    for residue in protein:
        for atom in residue[2]:
            if atom[0] == "CA":
                atoms.append(atom)
    return atoms

# Transforms the coords to the relative center of the protein
def center(protein):

    # Center of the protein
    centroid = [0, 0, 0]
    atoms = extractAtomCoords(protein)
    for atom in atoms:
        centroid += atom[1]

    try:
        centroid = centroid/len(atoms)
    except ZeroDivisionError as er:
        print("We tried to divide by zero. This is a run-time error: ", er)
        sys.exit(0)

    # Finds the transformed coordinates (relative to center of protein)
    newProtein = []
    for residue in protein:
        residueAtoms = []
        for atom in residue[2]:
            residueAtoms.append((atom[0], atom[1] - centroid))
        newProtein.append((residue[0], residue[1], residueAtoms))

    return newProtein


# Rotates atoms2 coordinates to minimize the RMSD value
def rotate(protein1, protein2):
    atoms1 = extractAtomCoords(protein1)
    atoms2 = extractAtomCoords(protein2)

    atom1Coords = []
    atom2Coords = []

    for atom1, atom2 in zip(atoms1, atoms2):
        atom1Coords.append(atom1[1])
        atom2Coords.append(atom2[1])

    coords1 = np.array(atom1Coords)
    coords2 = np.array(atom2Coords)

    assert(coords1.shape[1] == 3)
    assert(coords2.shape[1] == 3)

    h = np.dot(np.transpose(coords1), coords2)

    u, s, vh = np.linalg.svd(h)

    is_reflection = (np.linalg.det(u) * np.linalg.det(vh)) < 0.0

    if is_reflection:
        s[-1] = -s[-1]

    d = np.linalg.det(np.transpose(np.matmul(u, vh)))

    m = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, d]])

    # This is the rotation matrix now
    rotateMatrix = np.matmul(np.matmul(np.transpose(vh), m), np.transpose(u))

    newProtein = []
    i = 0
    for residue in protein2:
        residueAtoms = []
        for atom in residue[2]:
            residueAtoms.append((atom[0], np.matmul(np.transpose(coords2[i]), rotateMatrix)))
            i += 1
        newProtein.append((residue[0], residue[1], residueAtoms))

    return newProtein


# Calculates the RMSF from the structures and time interval given, returns a list of data
def calculateRMSF(structures, time):
    # find average position

    # Holds the average position for each atom
    refPosition = []

    # go over every atom in the structures
    for i in range(len(structures[0])):
        tempPosition = np.asarray([0.0, 0.0, 0.0])
        for step in structures:
            tempPosition += step[i][1]

        tempPosition = tempPosition / len(structures)
        refPosition.append(tempPosition)

    data = []

    for i in range(len(refPosition)):
        tempData = 0.0

        for step in structures:
            tempData += np.dot(step[i][1] - refPosition[i], step[i][1] - refPosition[i])

        tempData = tempData / time
        tempData = math.sqrt(tempData)
        data.append(tempData)

    return data

def main():

    # Find the QM region atoms
    qmdmdDocName = " "
    qmRawList = []

    filenames = os.listdir("../")
    for filename in filenames:
        if ".qmdmd" in filename:
            qmdmdDocName = "../"+filename
            break

    try:
        record = False
        with open(qmdmdDocName) as document:
            for line in document:
                if "QM Residues" in line:
                    record = True
                    continue
                elif "#" in line or "Custom Substrate" in line:
                    record = False
                else:
                    if record:
                        line = "".join(line.split())
                        qmRawList.append(line)
    except IOError:
        print("Could not open the QM list")
        sys.exit(0)

    # removes any blanklines
    qmRawList = list(filter(None, qmRawList))
    qmList = []
    for residue in qmRawList:
        qmList.append( (residue[0:3], int(residue[3:])))

    # Find the pdb files
    filenames = os.listdir(".")
    result = []
    for filename in filenames:
        if os.path.isdir(os.path.join(os.path.abspath("."), filename)):
            result.append(filename)

    i = 0
    while i < len(result):
        if "Iteration" not in result[i]:
            del result[i]
        else:
            result[i] = cwd + "/" + result[i]
            i += 1

    # Load in the PDB files into proteins data types
    proteins = []
    for iteration in result[1:]:
        filenames = os.listdir(iteration)
        if "winner.pdb" in filenames:
            iteration = iteration + "/winner.pdb"
        elif "to_next_iteration.pdb" in filenames:
            iteration = iteration + "/to_next_iteration.pdb"
        proteins.append(extractPDB(iteration))

    # Rotate each protein to minimize RMSD relative to starting PDB
    for index, protein in enumerate(proteins[1:]):
        proteins[index] = rotate(proteins[0], protein)


    # Calculates the RMSF data
    # Time is in nanoseconds here (QM/DMD is a 20ns simulation)
    # But we use total number of stuctures (1-40 here)
    allAtoms = []
    qmAtoms = []
    alphaCarbonsAtoms = []
    for protein in proteins[1:]:
        allAtoms.append(extractAtomCoords(protein))
        qmAtoms.append(extractQMAtoms(protein, qmList))
        alphaCarbonsAtoms.append(extractAlphaCarbons(protein))

    allAtomData = calculateRMSF(allAtoms, 40)
    qmAtomData = calculateRMSF(qmAtoms, 40)
    alphaCarbonData = calculateRMSF(alphaCarbonsAtoms, 40)

    # Write-out the data
    with open("RMSF_Data.csv", 'w') as f:
        f.write("Atom\n")
        for data in qmAtomData:
            f.write(", " + str(data))

        f.write("\n")
        for data in alphaCarbonData:
            f.write(", " + str(data))

        f.write("\n")
        for data in allAtomData:
            f.write(", " + str(data))


if __name__ == "__main__":
    main()
