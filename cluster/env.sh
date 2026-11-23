# Environment for Ancalagon jobs on the TAMU cyclotron condor pool.
# Sourced by run_ancalagon_job.sh; also usable interactively on c102.
# Everything lives on /data/pagroup/pa02 (the /home/pagroup quota is ~full,
# and condor's own /tmp is a shared 200G NFS that must not be filled).
export DRAGON_NN=/data/pagroup/pa02/padsley/dragon_nn
export ANCALAGON_BIN=$DRAGON_NN/Ancalagon/build/Ancalagon
export G4INSTALL=/home/pagroup/padsley/codes/geant4_11.3_install
export LD_LIBRARY_PATH=$G4INSTALL/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
G4D=$DRAGON_NN/geant4_11.3_data
export G4LEDATA=$G4D/G4EMLOW8.6.1
export G4LEVELGAMMADATA=$G4D/PhotonEvaporation6.1
export G4RADIOACTIVEDATA=$G4D/RadioactiveDecay6.1.2
export G4ABLADATA=$G4D/G4ABLA3.3
export G4INCLDATA=$G4D/G4INCL1.2
export G4ENSDFSTATEDATA=$G4D/G4ENSDFSTATE3.0
# G4NDL is not copied to /data: EM-only Ancalagon never reads it.
export G4NEUTRONHPDATA=/home/pagroup/padsley/codes/geant4_11.3_data/G4NDL4.7.1
export G4PARTICLEXSDATA=$G4D/G4PARTICLEXS4.1
export G4PIIDATA=$G4D/G4PII1.3
export G4SAIDXSDATA=$G4D/G4SAIDDATA2.0
export G4REALSURFACEDATA=$G4D/RealSurface2.2
export G4CHANNELINGDATA=$G4D/G4CHANNELING1.0
export NIDHOGG_PY=$DRAGON_NN/pyenv/bin/python
