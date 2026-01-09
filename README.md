# Machine-Learning-to-extract-intrinsic-parameters-for-a-PIC-platform-

This repository contains the code used to infer intrinsic physical parameters
(onsite losses and resonant frequency shifts) of coupled ring-resonator
photonic integrated circuits (PICs) from spectral measurements using
supervised machine learning.

The code accompanies an upcoming research manuscript and is organized by physical
“cases” (A, B, C), corresponding to increasing model complexity.

## Repository Structure
- data/
   - case_A/    - we have a sample space of 1d chain of 8 ring resonators with all having different onsite losses, but same resonant frequencies for each ring (contains frequencies.txt, experiemntal spectra as full.txt, and onsite_lossesinGHz_afteradding_intrinsicloss.txt as data for onsite losses)
   - case_B/    - we have a sample space of 1d chain of 8 ring resonators with all having different resonant frequencies, but same onsite losses for each ring (contains frequencies.txt, experiemntal spectra as full.txt, and finalized_resonant_freqs_wrt_ref_phase_using_fsr_and_puc_phases.txt as data for resonant frequencies)
   - case_C/    - we have a sample space of 1d chain of 8 ring resonators with all having different resonant frequencies and onsite losses for each ring (contains frequencies.txt, experiemntal spectra as full.txt, finalized_resonant_freqs_wrt_ref_phase_using_fsr_and_puc_phases.txt as data for resonant frequencies, and random_oniste_losses_inGHz.txt for onsite losses)

- src/
  - picml/
      -init.py   
      - dataio.py -contains functions to load input data
      - models.py  -contains the neural networks for all the cases
      - theory.py  - contains functions to generate theory data for the stage 2 of all the cases
      - utils.py  - contains necessary functions
      - training.py - contains the training routines for the stage 1 and stage 2 of all the three cases

- notebooks/ - Trains a neural network to extract Case A,B, or C respectively parameters from measured spectra. The predicted parameters are then used to generate theoretical spectra, which are compared directly with the experimental data. Each notebook contains the loading of data, training and prediction of neural networks, and finally a sorted figure to compare the predicted and true neural network.
  - case_A_execution.ipynb  
  - case_B_execution.ipynb
  - case_C_execution.ipynb

- environment.yml
- README.md

## Running the Code

1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate picml

---

## 7. Explicitly say what is *not* included (this helps you)

```markdown
## Notes

- The notebooks are intended as reproducible research scripts, not as a
  standalone software package.
- Automated tests are not included at this stage.
- Some datasets may be preprocessed or reduced versions of experimental data
  used in the associated manuscript.



