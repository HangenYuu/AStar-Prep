# GROMACS
GROMACS is an engine to perform molecular dynamics simulations and energy minimization.
# Course
- [https://web.stanford.edu/class/cs279/](https://tutorials.gromacs.org/md-intro-tutorial.html) - Quite basic lecture slides, but good for my case because I do need to brush up on the basic.
# Tutorial
- [https://tutorials.gromacs.org/md-intro-tutorial.html](https://tutorials.gromacs.org/md-intro-tutorial.html) - The most convenient tutorial online because GROMACS and PyMol turn out to be quite a pain in the \*ss to install. But this one comes with an online ready environment to run everything.
# Tutorial 1
A complete run of GROMACS using the Factor Xa (1FJS) structure pulled from Protein Data Bank
## Workflow
### System Preparation Phase
#### 1. Initial Structure Processing (`gmx pdb2gmx`)
- Remove unwanted atoms and predefined connectivity information.
```bash
grep -v HETATM input/1fjs.pdb > 1fjs_protein_tmp.pdb
grep -v CONECT 1fjs_protein_tmp.pdb > 1fjs_protein.pdb
```
- Check for missing entries which can break the simulations later
```bash
grep MISSING input/1fjs.pdb
```
- Convert PDB structure file to GROMACS-compatible format
- Generate three key files with appropriate force field and water model:
    - **Structure file (`.gro`)**: Coordinates in GROMACS format
    - **Topology file (`.top`)**: Contains force field parameters for each atom
    - **Position restraint file (`.itp`)**: Used during equilibration steps
```bash
gmx pdb2gmx -f 1fjs_protein.pdb -o 1fjs_processed.gro -water tip3p -ff "charmm27"
```

| Option   | Effect                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-water` | Water model to use: `none`, `spc`, `spce`, `tip3p`, `tip4p`, `tip5p`, `tips3p`.                                                                                                                                                                                                                                                                                                                            |
| `-ignh`  | Ignore H atoms in the PDB file; especially useful for NMR structures. Otherwise, if H atoms are present, they must be in the named exactly how the force fields in GROMACS expect them to be. Different conventions exist, so dealing with H atoms can occasionally be a headache! If you need to preserve the initial H coordinates, but renaming is required, then the Linux sed command is your friend. |
| `-ter`   | Interactively assign charge states for N- and C-termini.                                                                                                                                                                                                                                                                                                                                                   |
| `-inter` | Interactively assign charge states for Glu, Asp, Lys, Arg, and His; choose which Cys are involved in disulfide bonds.                                                                                                                                                                                                                                                                                      |

#### 2. Define Simulation Box (`gmx editconf`)
- Create a simulation box around the molecule
- Choose box dimensions and shape (rhombic dodecahedron is most efficient?)
- Center the structure within the box
```bash
gmx editconf -f 1fjs_processed.gro -o 1fjs_newbox.gro -c -d 1.0 -bt dodecahedron
```
#### 3. Solvation (`gmx solvate`)
- Add solvent molecules (typically water) around the solute
- Updates the topology file to include solvent information
- Creates a solvated system ready for further processing
```bash
gmx editconf -f 1fjs_processed.gro -o 1fjs_newbox.gro -c -d 1.0 -bt dodecahedron
```
> "spc216.gro" is a generic equilibrated 3-point solvent model box. You can use spc216.gro as the solvent configuration for SPC, SPC/E, or TIP3P water, since they are all three-point water models.
#### 4. Add Counterions (`gmx genion`)
- Neutralize the overall system charge by adding counterions
- Replace solvent molecules with appropriate ions (Na⁺, Cl⁻)
> To produce a .tpr file with `gmx grompp`, we will need an additional input file, with the extension .mdp (molecular dynamics parameter file); `gmx grompp` will assemble the parameters specified in the .mdp file with the coordinates and topology information to generate a .tpr file. It can just be an empty file for this case.
```bash
touch ions.mdp
gmx grompp -f ions.mdp -c 1fjs_solv.gro -p topol.top -o ions.tpr
```
> Many GROMACS tools promt for input at the command line. It was quite a hassle to run in a Jupyter Notebook at times. You usually need to anticipate the required input and feed it with a print command in a shell pipe.
```bash
printf "SOL\n" | gmx genion -s ions.tpr -o 1fjs_solv_ions.gro -conc 0.15 -p topol.top -pname NA -nname CL -neutral
```
### Simulation Phase
The simulation phase uses two main commands for all steps: **`gmx grompp`** (preparation) and **`gmx mdrun`** (execution).
#### 1. Energy Minimization
- **Purpose**: Remove bad contacts and optimize initial geometry
- **Process**: Use `grompp` to create run input file (`.tpr`), then `mdrun` to execute
- **Files needed**: Structure (`.gro`), topology (`.top`), parameters (`.mdp`). The `.mdp` file needs to be defined beforehand e.g.,
```bash
title       = CHARMM steepest descent enrgy minimisation

; Parameters describing what to do, when to stop and what to save
integrator  = steep  ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0 ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01   ; Minimization step size
nstenergy   = 500    ; save energies every 1.0 ps, so we can observe if we are successful
nsteps      = -1     ; run as long as we need
; Settings that make sure we run with parameters in harmony with the selected force-field
constraints             = h-bonds   ; bonds involving H are constrained
rcoulomb                = 1.2       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.2       ; short-range van der Waals cutoff (in nm)
vdw-modifier            = Force-switch ;  specific CHARMM
rvdw_switch             = 1.0       ;
DispCorr                = no        ; account for cut-off vdW scheme -
;in case of CHARMM DispCorr = EnerPres only for monolayers
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
fourierspacing          = 0.15     ; grid spacing for FFT
```
```bash
gmx grompp -f input/emin-charmm.mdp -c 1fjs_solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em -ntmpi 1 -ntomp 1
```
Export results
```bash
printf "Potential\n0\n" | gmx energy -f em.edr -o potential.xvg -xvg none
```
#### 2. NVT Equilibration (Constant Volume)
- **Purpose**: Equilibrate temperature while keeping volume constant
- **Uses position restraints** to prevent large structural changes. Achieve by adding `define = -DPOSRES` to the `.mdp` file.
```bash
title                   = CHARMM NVT equilibration 
define                  = -DPOSRES  ; position restrain the protein

; Parameters describing what to do, when to stop and what to save
integrator              = md        ; leap-frog integrator
[...]
```
- Other options
  - `gen_vel = yes`: Initiates velocity generation. Using different random seeds (gen_seed) gives different initial velocities, and thus multiple (different) simulations can be conducted from the same starting structure.
  - `tcoupl = V-rescale`: The velocity rescaling thermostat is an improvement upon the Berendsen weak coupling method, which did not reproduce a correct kinetic ensemble.
  - `pcoupl = no`: Pressure coupling is not applied.
```bash
gmx grompp -f input/nvt-charmm.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -ntmpi 2 -ntomp 8 -v -deffnm nvt
```
#### 3. NPT Equilibration (Constant Pressure)
- **Purpose**: Equilibrate both temperature and pressure
- **Allows system density** to adjust to target conditions
- **May continue using position restraints**
```bash
gmx grompp -f input/npt-charmm.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -ntmpi 2 -ntomp 8 -v -deffnm npt
```
#### 4. Production Run
- **Purpose**: Collect data for analysis
- **Remove position restraints** for free dynamics
- **Generate trajectory files** for subsequent analysis
- **Longest simulation step** - duration depends on research question
```bash
gmx grompp -f input/md-charmm.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
gmx mdrun -ntmpi 2 -ntomp 8 -v -deffnm md
```
> The general command is mostly unchanged between the 4 steps.
# Tutorial 2 - Simulation of a Membrane Protein using GROMACS
Using the maltoporin channel with PDB code 1MAL.
The tutorial made use of a GUI to generate the input files for GROMACS: https://zenodo.org/records/10794193

It's quite similar to the first one, with some additional notes:
- Simulation can be very time-consuming and storage intensive (150,000 steps, each for 1 picosecond, took several days and 9GB of data).
- When relaxing the positional restraints in NVT and NPT steps, it needs to be done in several steps to avoid sudden change in the system.
- The root mean square deviation (RMSD) is a practical parameter to compare the backbones of a protein from its initial to final state, which illustrates the dynamics of structure during the simulation.
# Tutorial 3 - Umbrella sampling
The tutorial for the foundational *conformational sampling* techniques. Use umbrella sampling to examine the free-energy profile of bringing two pyrimidine molecules together in an aqueous solution.

Standard MD simulation like above produces an ensemble of conformations that are likely at a given temperature using physics rules. However, the simulations struggle to account for rare events where the molecules have to cross a high energy barrier to move from one conformation to another conformation. To compensate for this instead of free sampling like the traditional MD simulation, we perform biased sampling by introducing artificial constraints to force the system to stay in/near a particular conformation.

Umbrella sampling is a basic method. It is performed by defining a reaction coordinates such as the distance between two molecules, then divide the coordinates into different conformation windows. In each window, we introduce an artificial energy potential that restricts the conformation of the molecules within that window despite the potential high energy of the conformation. The energy potential introduced is often harmonic, which leads to a curve that looks like an umbrella. Simulation is performed in each window separately, who is obsession and overlaps with each other, leading to a complete profile. The biases distort the natural probability. Hence, the biased energy potentials need to be combined using specific techniques such as Weighted Histogram Analysis Method (WHAM) to remove the biased energy. The final result is an unbiased energy profile covering a full energy landscape of stable conformations and barriers.

In GROMACS, each window run is a separate MD run, following the steps above