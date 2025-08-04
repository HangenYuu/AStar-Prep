# GROMACS
GROMACS is an engine to perform molecular dynamics simulations and energy minimization.
# Course
- [https://web.stanford.edu/class/cs279/](https://tutorials.gromacs.org/md-intro-tutorial.html) - Quite basic lecture slides, but good for my case because I do need to brush up on the basic.
# Tutorial
- [https://tutorials.gromacs.org/md-intro-tutorial.html](https://tutorials.gromacs.org/md-intro-tutorial.html) - The most convenient tutorial online because GROMACS and PyMol turn out to be quite a pain in the \*ss to install. But this one comes with an online ready environment to run everything.
# Tutorial 1
A complete run of GROMACS using a random structure pulled from Protein Data Bank
## Workflow
### System Preparation Phase
#### 1. Initial Structure Processing (`gmx pdb2gmx`)
- Convert PDB structure file to GROMACS-compatible format
- Generate three key files:
    - **Topology file (`.top`)**: Contains force field parameters for each atom
    - **Position restraint file (`.itp`)**: Used during equilibration steps
    - **Structure file (`.gro`)**: Coordinates in GROMACS format
- Select appropriate force field and water model
#### 2. Define Simulation Box (`gmx editconf`)
- Create a simulation box around the molecule
- Choose box dimensions and shape (rhombic dodecahedron is most efficient)
- Center the structure within the box
#### 3. Solvation (`gmx solvate`)
- Add solvent molecules (typically water) around the solute
- Updates the topology file to include solvent information
- Creates a solvated system ready for further processing
#### 4. Add Counterions (`gmx genion`)
- Neutralize the overall system charge by adding counterions
- Replace solvent molecules with appropriate ions (Na⁺, Cl⁻)
### Simulation Phase
The simulation phase uses two main commands for all steps: **`gmx grompp`** (preparation) and **`gmx mdrun`** (execution).
#### 1. Energy Minimization
- **Purpose**: Remove bad contacts and optimize initial geometry
- **Process**: Use `grompp` to create run input file (`.tpr`), then `mdrun` to execute
- **Files needed**: Structure (`.gro`), topology (`.top`), parameters (`.mdp`)
#### 2. NVT Equilibration (Constant Volume)
- **Purpose**: Equilibrate temperature while keeping volume constant
- **Duration**: Typically short equilibration period
- **Uses position restraints** to prevent large structural changes
#### 3. NPT Equilibration (Constant Pressure)
- **Purpose**: Equilibrate both temperature and pressure
- **Allows system density** to adjust to target conditions
- **May continue using position restraints**
#### 4. Production Run
- **Purpose**: Collect data for analysis
- **Remove position restraints** for free dynamics
- **Generate trajectory files** for subsequent analysis
- **Longest simulation step** - duration depends on research question
# Tutorial 2
